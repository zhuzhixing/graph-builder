import os
import re
import sys
import numpy as np
import pandas as pd
import ftplib
import json
import wget
import datetime
import requests
import logging
from pathlib import Path
from multiprocessing import Pool
from typing import Type, List, Tuple, Optional
from dataclasses import dataclass


logger = logging.getLogger("graph_builder.parsers.base_parser")


def get_matched_entity_id(args: Tuple[str, dict]):
    i, entity_type_ids = args
    matched = entity_type_ids.get(i, None)
    if matched is not None:
        return matched
    else:
        return None


def get_value(v, label) -> str:
    labels = {
        "entity_id": 0,
        "entity_type": 1,
    }

    if v is not None:
        return v[labels[label]]
    else:
        return ""


def generate_entity_type_id(entity_id, entity_type):
    return str(entity_id) + "#" + str(entity_type)


def generate_source_key(row):
    return str(row["source_id"]) + "#" + str(row["source_type"])


def generate_target_key(row):
    return str(row["target_id"]) + "#" + str(row["target_type"])


@dataclass
class Download:
    download_url: str
    filename: str
    is_downloadable: bool = True


@dataclass
class BaseConfig:
    downloads: List[Download]
    database: str


@dataclass
class Relation:
    source_id: str
    target_id: str
    source_type: str
    target_type: str
    relation_type: str
    resource: str
    pmids: Optional[str]
    key_sentence: Optional[str]

    @classmethod
    def from_args(cls, **kwargs):
        try:
            initializer = cls.__initializer
        except AttributeError:
            # Store the original init on the class in a different place
            cls.__initializer = initializer = cls.__init__
            # replace init with something harmless
            cls.__init__ = lambda *a, **k: None

        # code from adapted from Arne
        added_args = {}
        for name in list(kwargs.keys()):
            if name not in cls.__annotations__:
                added_args[name] = kwargs.pop(name)

        ret = object.__new__(cls)
        initializer(ret, **kwargs)  # type: ignore
        # ... and add the new ones by hand
        for new_name, new_val in added_args.items():
            setattr(ret, new_name, new_val)

        return ret


class BaseParser:
    def __init__(
        self,
        reference_entity_file: Path,
        db_directory: Path,
        output_directory: Path,
        config: BaseConfig | None = None,
        download=True,
        skip=True,
        num_workers: int = 20,
        relation_type_dict_df: pd.DataFrame | None = None,
    ):
        if config is None:
            raise ValueError("config is required, you need to specify it in subclass.")

        self.output_directory = output_directory.joinpath(config.database)
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        self.db_directory = db_directory.joinpath(config.database)
        if not os.path.exists(self.db_directory):
            os.makedirs(self.db_directory)

        self.config = config
        self.download = download

        if self.download:
            failed_downloads = []
            for download in self.config.downloads:
                if download.is_downloadable:
                    self.download_db(
                        download.download_url,
                        self.db_directory,
                        download.filename,
                        skip=skip,
                    )
                else:
                    if not os.path.exists(self.db_directory / download.filename):
                        logger.warning(
                            "Cannot download data automatically, please access the '%s' url and download %s file manually. After downloaded, you need to place it into '%s' directory."
                            % (
                                download.download_url,
                                download.filename,
                                self.db_directory,
                            )
                        )
                        failed_downloads.append(download.filename)
                    else:
                        logger.info(
                            "Found %s file in %s directory, skip to download it."
                            % (download.filename, self.db_directory)
                        )

            if len(failed_downloads) > 0:
                sys.exit(1)

        self.entities = self._read_reference_entity_file(reference_entity_file)

        # For compatibility with the previous version
        # In future, we will remove the "SideEffect" label from the entities file and treat the side effect as a relation type between drugs and diseases.
        if "SideEffect" not in self.entities["label"].tolist():
            self.treat_side_effect_as_disease = True
        else:
            self.treat_side_effect_as_disease = False

        self.num_workers = num_workers
        self.relation_type_dict_df = relation_type_dict_df

    @property
    def raw_filepaths(self) -> List[Path]:
        """Returns all the raw filepaths which specified in the config. You can get and use them in the subclass.

        Returns:
            List[Path]: List of raw filepaths.
        """
        return [
            self.db_directory / download.filename for download in self.config.downloads
        ]

    @property
    def database(self):
        """Returns the database name which is used to generate the output directory and output filename."""
        return self.config.database

    @property
    def output_filepath(self) -> Path:
        """Returns the output filepath. We will save the formatted relations in this file. Because we expect to save all relations into a single file, so you need to implement the `extract_relations` method to extract all relations from the database file and return a list of Relation objects. Then, we will convert the list of Relation objects to a dataframe and save it into this file.

        Returns:
            Path: A Path object which points to the output file.
        """
        filepath = self.output_directory / f"formatted_{self.database}.tsv"
        return filepath

    @staticmethod
    def _read_reference_entity_file(filepath: Path):
        """Reads the reference entity file as a pandas dataframe.

        Args:
            filepath (Path): Path to the reference entity file.

        Raises:
            FileNotFoundError: If the reference entity file is not found.

        Returns:
            DataFrame: Pandas dataframe containing all the reference entities.
        """
        if filepath.exists():
            return pd.read_csv(filepath, sep="\t")
        else:
            raise FileNotFoundError("The reference entity file is not found.")

    @staticmethod
    def download_from_ftp(ftp_url, user, password, to, file_name):
        try:
            domain = ftp_url.split("/")[2]
            ftp_file = "/".join(ftp_url.split("/")[3:])
            with ftplib.FTP(domain) as ftp:
                ftp.login(user=user, passwd=password)
                with open(os.path.join(to, file_name), "wb") as fp:
                    ftp.retrbinary("RETR " + ftp_file, fp.write)
        except ftplib.error_reply as err:
            raise ftplib.error_reply(
                "Exception raised when an unexpected reply is received from the server. {}.\nURL:{}".format(
                    err, ftp_url
                )
            )
        except ftplib.error_temp as err:
            raise ftplib.error_temp(
                "Exception raised when an error code signifying a temporary error. {}.\nURL:{}".format(
                    err, ftp_url
                )
            )
        except ftplib.error_perm as err:
            raise ftplib.error_perm(
                "Exception raised when an error code signifying a permanent error. {}.\nURL:{}".format(
                    err, ftp_url
                )
            )
        except ftplib.error_proto as err:
            raise ftplib.error_proto(
                "Exception raised when a reply is received from the server that does not fit the response specifications of the File Transfer Protocol. {}.\nURL:{}".format(
                    err, ftp_url
                )
            )

    def download_db(
        self,
        database_url,
        directory,
        file_name=None,
        user="",
        password="",
        avoid_wget=False,
        skip=True,
    ):
        """
        This function downloads the raw files from a biomedical database server when a link is provided.

        :param str databaseURL: link to access biomedical database server.
        :param directory:
        :type directory: str or None
        :param file_name: name of the file to dowload. If None, 'databaseURL' must contain \
                            filename after the last '/'.
        :type file_name: str or None
        :param str user: username to access biomedical database server if required.
        :param str password: password to access biomedical database server if required.
        """
        if file_name is None:
            file_name = database_url.split("/")[-1].replace("?", "_").replace("=", "_")

        header = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
        }

        filepath = os.path.join(directory, file_name)
        logger.info("Download file from %s into %s" % (database_url, filepath))
        if os.path.exists(filepath):
            if skip:
                logger.info("%s exists, don't need to download it." % filepath)
                return None
            else:
                os.remove(filepath)
        try:
            if database_url.startswith("ftp:"):
                self.download_from_ftp(
                    database_url, user, password, directory, file_name
                )
            else:
                try:
                    if not avoid_wget:
                        wget.download(database_url, os.path.join(directory, file_name))
                    else:
                        r = requests.get(database_url, headers=header)
                        with open(os.path.join(directory, file_name), "wb") as out:
                            out.write(r.content)
                except Exception:
                    r = requests.get(database_url, headers=header)
                    with open(os.path.join(directory, file_name), "wb") as out:
                        out.write(r.content)
            # time.sleep(10)
        except Exception as err:
            raise Exception(
                "Something went wrong. {}.\nURL:{}".format(err, database_url)
            )

    @staticmethod
    def list_directory_files(directory):
        """
        Lists all files in a specified directory.

        :param str directory: path to folder.
        :return: List of file names.
        """
        from os import listdir
        from os.path import isfile, join

        onlyfiles = [
            f
            for f in listdir(directory)
            if isfile(join(directory, f)) and not f.startswith(".")
        ]

        return onlyfiles

    @staticmethod
    def list_directory_folders(directory):
        """
        Lists all directories in a specified directory.

        :param str directory: path to folder.
        :return: List of folder names.
        """
        from os import listdir
        from os.path import isdir, join

        dircontent = [
            f
            for f in listdir(directory)
            if isdir(join(directory, f)) and not f.startswith(".")
        ]
        return dircontent

    def get_current_time(self):
        """
        Returns current date (Year-Month-Day) and time (Hour-Minute-Second).

        :return: Two strings: date and time.
        """
        now = datetime.datetime.now()
        return "{}-{}-{}".format(now.year, now.month, now.day), "{}:{}:{}".format(
            now.hour, now.minute, now.second
        )

    def file_size(self, file_path):
        """
        This function returns the file size.

        :param str file_path: path to file.
        :return: Size in bytes of a plain file.
        :rtype: str
        """
        if os.path.isfile(file_path):
            file_info = os.stat(file_path)
            return str(file_info.st_size)

    def _build_stats(self, count, otype, name, dataset, filename, updated_on=None):
        """
        Returns a tuple with all the information needed to build a stats file.

        :param int count: number of entities/relations.
        :param str otype: 'entity' or 'relationsgips'.
        :param str name: entity/relation label.
        :param str dataset: database/ontology.
        :param str filename: path to file where entities/relations are stored.
        :return: Tuple with date, time, database name, file where entities/relations are stored, \
        file size, number of entities/relations imported, type and label.
        """
        y, t = self.get_current_time()
        size = self.file_size(filename)
        filename = filename.split("/")[-1]

        return (str(y), str(t), dataset, filename, size, count, otype, name, updated_on)

    def build_stats(self):
        raise NotImplementedError("build_stats method is not implemented.")

    def extract_relations(self) -> List[Relation]:
        """Extracts relations from the database file and returns a list of Relation objects.

        Raises:
            NotImplementedError

        Returns:
            List[Relation]: List of Relation objects.
        """
        raise NotImplementedError("extract_relations method is not implemented.")

    def parse(self) -> List[Relation]:
        """Parses all relations and convert the source ids and target ids to the standard format.

        Returns:
            List[Relation]: List of Relation objects.
        """
        relations = self.extract_relations()
        logger.info("Found %s relations." % len(relations))
        entity_id_map = self.get_entity_id_maps(relations, self.num_workers)
        logger.info("Found %s entity ids in entity id map." % len(entity_id_map.keys()))

        # Convert a list of relations to a dataframe
        df = pd.DataFrame(relations, dtype=str)

        # Dropna method cannot identify empty string as NaN, so we need to replace empty string with None
        df.replace("", None, inplace=True)

        if self.treat_side_effect_as_disease:
            # If the side effect is not in the entities file, we treat it as a disease
            df["source_type"].replace("SideEffect", "Disease", inplace=True)
            df["target_type"].replace("SideEffect", "Disease", inplace=True)

        # Remove all rows with empty source_id or target_id
        logger.info("The number of relations before dropna: %s" % len(df))
        df.dropna(subset=["source_id", "target_id"], inplace=True)
        logger.info("The number of relations after dropna: %s" % len(df))
        output_file = self.output_directory / f"raw_{self.database}.tsv"
        df.to_csv(output_file, sep="\t", index=False)

        df["combined_key"] = df.apply(generate_source_key, axis=1)
        df_dict = {key: group_df for key, group_df in df.groupby("combined_key")}

        for idx, key in enumerate(df_dict.keys()):
            logger.info("Processing %.2f%%" % ((idx * 1.0 / len(df_dict.keys())) * 100))
            value = entity_id_map.get(key, None)
            df_dict[key]["source_entity_id"] = get_value(value, "entity_id")
            df_dict[key]["source_entity_type"] = get_value(value, "entity_type")

        # Restore the dataframe
        df = pd.concat(df_dict.values())

        df["combined_key"] = df.apply(generate_target_key, axis=1)
        df_dict = {key: group_df for key, group_df in df.groupby("combined_key")}
        for idx, key in enumerate(df_dict.keys()):
            logger.info("Processing %.2f%%" % ((idx * 1.0 / len(df_dict.keys())) * 100))
            value = entity_id_map.get(key, None)
            df_dict[key]["target_entity_id"] = get_value(value, "entity_id")
            df_dict[key]["target_entity_type"] = get_value(value, "entity_type")

        # Restore the dataframe
        df = pd.concat(df_dict.values())

        logger.info("The number of relations: %s" % len(df))
        # df.dropna(inplace=True)
        # logger.info("The number of relations after dropna: %s" % len(df))

        df.rename(
            columns={
                "source_id": "raw_source_id",
                "source_type": "raw_source_type",
                "source_entity_id": "source_id",
                "source_entity_type": "source_type",
                "target_id": "raw_target_id",
                "target_type": "raw_target_type",
                "target_entity_id": "target_id",
                "target_entity_type": "target_type",
            },
            inplace=True,
        )

        # Remove the combined key
        logger.info("Remove the combined_key column.")
        df.drop(columns=["combined_key"], inplace=True)

        if self.relation_type_dict_df is not None:
            relation_type_dict_df = self.relation_type_dict_df[
                ["relation_type", "formatted_relation_type"]
            ]
            logger.info("Replace the relation type with the formatted relation type.")

            df = df.merge(
                relation_type_dict_df,
                how="left",
                left_on="relation_type",
                right_on="relation_type",
            )
        else:
            logger.info("The relation type dictionary is not provided, skip to replace the relation type with the formatted relation type.")
            df["formatted_relation_type"] = df["relation_type"]

        def check_and_swap(row):
            if row["formatted_relation_type"] is not None:
                source_type_in_relation, target_type_in_relation = (
                    row["formatted_relation_type"].split("::")[2].split(":")
                )
            else:
                source_type_in_relation, target_type_in_relation = "", ""

            if source_type_in_relation == row["target_type"] and target_type_in_relation == row["source_type"]:
                row["source_id"], row["target_id"] = row["target_id"], row["source_id"]
                row["source_type"], row["target_type"] = row["target_type"], row["source_type"]

            if self.treat_side_effect_as_disease:
                # The relation type might be Hetionet::CcSE::Compound:SideEffect, we need to replace the SideEffect with Disease and the relation type should be Hetionet::CcSE::Compound:Disease when we would like to treat the side effect as a disease.
                # NOTE: We have replaced the SideEffect with Disease in the entity type in previous step. It's an essential step to make sure the source_type and target_type are consistent with the relation type.
                if source_type_in_relation == "SideEffect" or target_type_in_relation == "SideEffect":
                    row["relation_type"] = re.sub(
                        r"::SideEffect:", "::Disease:", row["relation_type"]
                    )
                    row["relation_type"] = re.sub(
                        r":SideEffect", ":Disease", row["relation_type"]
                    )
                    row["formatted_relation_type"] = re.sub(
                        r"::SideEffect:", "::Disease:", row["formatted_relation_type"]
                    )
                    row["formatted_relation_type"] = re.sub(
                        r":SideEffect", ":Disease", row["formatted_relation_type"]
                    )

            return row

        logger.info("Check and swap source_id and target_id if needed.")
        df = pd.DataFrame(df.apply(check_and_swap, axis=1))

        num_df = len(df)
        logger.info("The number of relations before filter_na: %s" % num_df)
        # source_id and target_id should not be empty
        empty_df = df[(df["source_id"] == "") | (df["target_id"] == "")]
        df = df[(df["source_id"] != "") & (df["target_id"] != "")]

        unformatted_outfile = self.output_directory / f"unformatted_{self.database}.tsv"
        logger.info(
            "Found %s invalid relations, all invalid relations are saved in %s"
            % (num_df - len(df), unformatted_outfile)
        )

        # Save all unformatted entities into a valid entity file
        # We want to build a knowledge graph with unformatted + formatted entities and relations, so we need to save all unformatted entities into a valid entity file.
        unformatted_source_entities = empty_df[empty_df["source_id"] == ""][
            ["raw_source_id", "raw_source_type"]
        ].drop_duplicates()
        # Rename the columns
        unformatted_source_entities.rename(
            columns={
                "raw_source_id": "id",
                "raw_source_type": "label",
            },
            inplace=True,
        )

        unformatted_target_entities = empty_df[empty_df["target_id"] == ""][
            ["raw_target_id", "raw_target_type"]
        ].drop_duplicates()
        # Rename the columns
        unformatted_target_entities.rename(
            columns={
                "raw_target_id": "id",
                "raw_target_type": "label",
            },
            inplace=True,
        )

        unformatted_entities = pd.concat(
            [unformatted_source_entities, unformatted_target_entities]
        ).drop_duplicates()

        # Add new columns
        unformatted_entities["name"] = unformatted_entities["id"]
        unformatted_entities["resource"] = "Unformatted" + self.database.title()
        unformatted_entities["description"] = ""
        unformatted_entities["synonyms"] = ""
        unformatted_entities["pmids"] = ""
        unformatted_entities["taxid"] = ""
        unformatted_entities["xrefs"] = ""

        # Save the unformatted entities
        unformatted_entity_outfile = (
            self.output_directory / f"unformatted_{self.database}_entities.tsv"
        )
        unformatted_entities.to_csv(
            unformatted_entity_outfile,
            sep="\t",
            index=False,
        )

        logger.info(
            "Found %s unformatted entities and they are saved in %s"
            % (len(unformatted_entities), unformatted_entity_outfile)
        )

        # Save all unformatted relations into a valid relation file
        # Fill the empty source_id and target_id with the raw_source_id and raw_target_id
        empty_df["source_id"].replace("", np.nan, inplace=True)
        empty_df["source_id"] = empty_df["source_id"].fillna(empty_df["raw_source_id"])
        empty_df["target_id"].replace("", np.nan, inplace=True)
        empty_df["target_id"] = empty_df["target_id"].fillna(empty_df["raw_target_id"])
        empty_df["source_type"].replace("", np.nan, inplace=True)
        empty_df["source_type"] = empty_df["source_type"].fillna(empty_df["raw_source_type"])
        empty_df["target_type"].replace("", np.nan, inplace=True)
        empty_df["target_type"] = empty_df["target_type"].fillna(empty_df["raw_target_type"])

        # Save the unformatted relations
        empty_df.to_csv(
            unformatted_outfile,
            sep="\t",
            index=False,
        )

        # Remove the deduplicated rows
        logger.info("The number of relations before drop_duplicates: %s" % len(df))
        # Save the deduplicated relations
        duplicated_relations = df[
            df.duplicated(
                subset=[
                    "source_id",
                    "source_type",
                    "target_id",
                    "target_type",
                    "relation_type",
                ],
                keep=False,
            )
        ]
        df.drop_duplicates(
            subset=[
                "source_id",
                "source_type",
                "target_id",
                "target_type",
                "relation_type",
            ],
            inplace=True,
        )

        duplicated_outfile = self.output_directory / f"duplicated_{self.database}.tsv"
        duplicated_relations.to_csv(
            duplicated_outfile,
            sep="\t",
            index=False,
        )
        logger.info(
            "Found %s duplicated relations and they are saved in %s"
            % (len(duplicated_relations), duplicated_outfile)
        )

        df.to_csv(
            self.output_filepath,
            sep="\t",
            index=False,
        )

        logger.info(
            "Finally, save %s relations in %s" % (df.shape[0], self.output_filepath)
        )

        return [Relation.from_args(**row) for row in df.to_dict("records")]  # type: ignore

    def get_entity_id_maps(self, relations: List[Relation], num_workers=20) -> dict:
        """
        Converts .tsv file with complete list of entity identifiers and aliases, \
        to dictionary with aliases as keys and entity identifiers as values.

        :param str relations: list of Relation objects.
        :return: Dictionary of f'{entity_type}#{entity_id}' (key) and related [entity_id, entity_type] (value).
        """
        outputfile = self.output_directory / f"{self.database}.entity_id_map.json"

        if os.path.exists(outputfile):
            logger.info(
                "Found entity id map file, skip to generate it. If you want to regenerate it, please delete the file: %s"
                % outputfile
            )
            with open(outputfile, "r") as fp:
                return json.load(fp)
        else:
            entity_type_ids = {}
            logger.info("Start to get entity id map.")
            for entity in self.entities.to_dict("records"):
                label = entity["label"]
                id = entity["id"]
                entity_type_id = generate_entity_type_id(id, label)
                entity_type_ids[entity_type_id] = [id, label]

                xrefs = entity["xrefs"]
                if xrefs and type(xrefs) == str:
                    for xref in xrefs.split("|"):
                        entity_type_id = generate_entity_type_id(xref, label)
                        entity_type_ids[entity_type_id] = [id, label]

            # Convert a list of relations to a dataframe
            df = pd.DataFrame(relations)

            source_entity_type_id = [
                "#".join(i)
                for i in zip(df["source_id"].tolist(), df["source_type"].tolist())
            ]

            deduped_source_entity_type_id = list(set(source_entity_type_id))

            target_entity_type_id = [
                "#".join(i)
                for i in zip(df["target_id"].tolist(), df["target_type"].tolist())
            ]

            deduped_target_entity_type_id = list(set(target_entity_type_id))

            deduped_entity_type_id = list(
                set(deduped_source_entity_type_id + deduped_target_entity_type_id)
            )

            # Create argument tuples for each call of get_matched_entity_id
            args = [(i, entity_type_ids) for i in deduped_entity_type_id]

            # Create a pool of workers
            with Pool(num_workers) as p:
                target_results = p.map(get_matched_entity_id, args)

            entity_id_map = dict(zip(deduped_entity_type_id, target_results))
            logger.info("The number of entity ids: %s" % len(entity_id_map.keys()))

            with open(outputfile, "w") as fp:
                json.dump(entity_id_map, fp)

            return entity_id_map
