from pathlib import Path
import pandas as pd
import logging
import re
from typing import List
from graph_builder.parsers.base_parser import BaseConfig, BaseParser, Relation, Download

logger = logging.getLogger("graph_builder.parsers.hsdn_parser")


class HsdnParser(BaseParser):
    """A parser for HSDN database. See https://www.nature.com/articles/ncomms5212 for more details."""

    def __init__(
        self,
        reference_entity_file: Path,
        db_directory: Path,
        output_directory: Path,
        download=True,
        skip=True,
        num_workers: int = 20,
        relation_type_dict_df=None,
    ):
        download_obj = Download(
            download_url="https://static-content.springer.com/esm/art%3A10.1038%2Fncomms5212/MediaObjects/41467_2014_BFncomms5212_MOESM1045_ESM.txt",
            filename="hsdn.tsv",
        )

        mesh_download_obj = Download(
            download_url="https://data.bioontology.org/ontologies/MESH/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb&download_format=csv",
            filename="MESH.csv.gz",
        )

        config = BaseConfig(
            downloads=[download_obj, mesh_download_obj],
            database="hsdn",
        )

        super().__init__(
            reference_entity_file,
            db_directory,
            output_directory,
            config,
            download,
            skip,
            num_workers,
            relation_type_dict_df,
        )

    def read_hsdn(self, hsdn_filepath: Path) -> pd.DataFrame:
        # Specify the column names
        df = pd.read_csv(
            hsdn_filepath,
            sep="\t",
            dtype=str,
        )

        df = df.rename(
            columns={
                "MeSH Symptom Term": "symptom",
                "MeSH Disease Term": "disease",
                "PubMed occurrence": "pubmed_occurrence",
                "TFIDF score": "tfidf_score",
            }
        )

        return df

    def get_value(self, row: pd.Series, column: str) -> str:
        item = row[column].values.tolist()
        if len(item) == 0:
            return ""
        else:
            return item[0]

    def read_mesh(self, mesh_filepath: Path) -> pd.DataFrame:
        # Read a csv.gz file and convert it to a data frame
        df = pd.read_csv(
            mesh_filepath,
            compression="gzip",
            header=0,
            sep=",",
            quotechar='"',
            dtype=str,
        )

        # Fill the nan values with empty string
        df = df.fillna("")

        # Select only the columns that we need
        df = df[
            ["Class ID", "Preferred Label", "Synonyms", "Definitions", "Semantic Types"]
        ]

        # Format the id column by using regex to replace the ".*MESH/" prefix with "MESH:". Must use the regex pattern
        df.loc[:, "Class ID"] = df["Class ID"].apply(
            lambda x: re.sub(r".*MESH/", "MESH:", x)
        )

        # Rename the columns
        df = df.rename(
            columns={
                "Class ID": "id",
                "Preferred Label": "name",
                "Synonyms": "synonyms",
                "Definitions": "description",
                "Semantic Types": "semantic_types",
            }
        )

        return df

    def format_df(self, df: pd.DataFrame, mesh: pd.DataFrame) -> pd.DataFrame:
        diseases = list(set(df["disease"]))
        disease_id_map = {
            x: self.get_value(mesh[mesh["name"] == x], "id") for x in diseases
        }

        symptoms = list(set(df["symptom"]))
        symptom_id_map = {
            x: self.get_value(mesh[mesh["name"] == x], "id") for x in symptoms
        }

        logger.info("Get id for %d symptoms" % len(symptom_id_map.keys()))
        df["target_id"] = list(map(lambda x: symptom_id_map[x], df["symptom"]))
        df["target_type"] = "Symptom"
        df["target_name"] = df["symptom"]

        logger.info("Get id for %d diseases" % len(disease_id_map.keys()))
        df["source_id"] = list(map(lambda x: disease_id_map[x], df["disease"]))
        df["source_type"] = "Disease"
        df["source_name"] = df["disease"]

        df["relation_type"] = "HSDN::has_symptom:Disease:Symptom"
        df["resource"] = "HSDN"
        df["key_sentence"] = ""
        df["pmids"] = ""

        df["score"] = df["tfidf_score"]

        df = df[
            [
                "source_id",
                "source_type",
                "target_id",
                "target_type",
                "relation_type",
                "resource",
                "key_sentence",
                "pmids",
                # Additional columns
                "source_name",
                "target_name",
                "score",
            ]
        ]

        return df

    def extract_relations(self) -> List[Relation]:
        raw_filepath = self.raw_filepaths[0]
        mesh_filepath = self.raw_filepaths[1]

        logger.info(f"Read {raw_filepath}")
        hsdn = self.read_hsdn(raw_filepath)

        logger.info(f"Read {mesh_filepath}")
        mesh = self.read_mesh(mesh_filepath)

        logger.info("Format the data frame...")
        hsdn = self.format_df(hsdn, mesh)

        logger.info("Warning: HSDN don't provide the ids for symptoms and diseases, so we use the names to match the ids. But this may cause some missing entities and relations.")

        logger.info("Get %d relations" % len(hsdn))

        return [Relation.from_args(**row) for row in hsdn.to_dict(orient="records")]  # type: ignore


if __name__ == "__main__":
    import logging
    import coloredlogs
    import verboselogs

    logging.basicConfig(level=logging.DEBUG)
    verboselogs.install()
    # Use the logger name instead of the module name
    coloredlogs.install(
        fmt="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
    )

    parser = HsdnParser(
        Path("/Volumes/ProjectData/Current/Datasets/biomedgps/graph_data/entities.tsv"),
        Path("/Users/jy006/Downloads/Development/biomedgps"),
        Path("/Users/jy006/Downloads/Development/biomedgps"),
    )

    parser.parse()
