from pathlib import Path
import pandas as pd
import logging
from typing import List
from graph_builder.parsers.base_parser import BaseConfig, BaseParser, Relation, Download

logger = logging.getLogger("graph_builder.parsers.drkg_parser")


class DrkgParser(BaseParser):
    def __init__(
        self,
        reference_entity_file: Path,
        db_directory: Path,
        output_directory: Path,
        download=True,
        skip=True,
        num_workers: int = 20,
    ):
        download_obj = Download(
            download_url="https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz",
            filename="drkg.tar.gz",
        )

        config = BaseConfig(
            downloads=[download_obj],
            database="drkg",
        )

        super().__init__(
            reference_entity_file,
            db_directory,
            output_directory,
            config,
            download,
            skip,
            num_workers,
        )

    def _extract_tar_gz(self, filepath: Path):
        import tarfile

        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=self.db_directory)

    def read_drkg(self, drkg_filepath: Path) -> pd.DataFrame:
        # Specify the column names
        df = pd.read_csv(
            drkg_filepath,
            sep="\t",
            header=None,
            dtype=str,
        )

        df = df.rename(
            columns={
                0: "source",
                1: "relation",
                2: "target",
            }
        )

        return df

    def format_id(self, entity_type: str, id: str) -> str | List[str]:
        if entity_type == "Gene":
            try:
                id_int = int(id)
                return f"ENTREZ:{id_int}"
            except ValueError:
                # Keep all the unknown gene id
                return id
        elif entity_type == "Compound":
            if id.startswith("DB"):
                return f"DrugBank:{id}"
            elif id.startswith("chebi"):
                return id.replace("chebi", "CHEBI")
            elif id.startswith("CHEMBL"):
                return f"CHEMBL:{id}"
            elif id.startswith("pubchem"):
                return id.replace("pubchem", "PUBCHEM")
            elif id.startswith("MESH"):
                return id
            elif id.startswith("hmdb"):
                return id.replace("hmdb", "HMDB")
            else:
                # TODO: How to deal with unknown compound id?
                return id
        elif entity_type == "Anatomy":
            # All id starts with UBERON
            return id
        elif entity_type == "Disease":
            # All disease id matches the pattern <database>:\d+, if not, we don't need to care about it. All unknown disease id will be filtered out at the end.
            return id
        elif entity_type == "SideEffect":
            if id.startswith("C"):
                return f"UMLS:{id}"
            else:
                return id
        elif entity_type == "PharmacologicClass":
            if id.startswith("N"):
                return f"NDF-RT:{id}"
            else:
                return id
        elif entity_type == "BiologicalProcess":
            return id
        elif entity_type == "CellularComponent":
            return id
        elif entity_type == "MolecularFunction":
            return id
        elif entity_type == "Pathway":
            if id.startswith("WP"):
                return f"WikiPathways:{id.split('_')[0]}"
            else:
                return id
        elif entity_type == "Symptom":
            if id.startswith("D"):
                return f"MESH:{id}"
            else:
                return id
        else:
            return id

    @staticmethod
    def _format_item(df: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
        df["source"] = df["source"].apply(lambda x: x.replace(source, target))
        df["target"] = df["target"].apply(lambda x: x.replace(source, target))
        df["relation"] = df["relation"].apply(lambda x: x.replace(source, target))

        return df

    def extract_relations(self) -> List[Relation]:
        raw_filepath = self.raw_filepaths[0]
        # Untar the file
        logger.info(f"Untar {raw_filepath}")
        self._extract_tar_gz(raw_filepath)
        drkg_filepath = self.db_directory / "drkg.tsv"

        logger.info(f"Read {drkg_filepath}")
        drkg = self.read_drkg(drkg_filepath)

        # Replace Side Effect with SideEffect, Pharmacologic Class with PharmacologicClass, Biological Process with BiologicalProcess, Cellular Component with CellularComponent, Molecular Function with MolecularFunction
        for entity_type in [
            "Side Effect",
            "Pharmacologic Class",
            "Biological Process",
            "Cellular Component",
            "Molecular Function",
        ]:
            drkg = self._format_item(drkg, entity_type, entity_type.replace(" ", ""))

        sep = "::"
        drkg["source_type"] = drkg["source"].apply(lambda x: x.split(sep)[0])
        drkg["target_type"] = drkg["target"].apply(lambda x: x.split(sep)[0])

        logger.info("Format the source id")
        drkg["source_id"] = drkg["source"].apply(
            lambda x: self.format_id(x.split(sep)[0], x.split(sep)[1])
        )

        logger.info("Format the target id")
        drkg["target_id"] = drkg["target"].apply(
            lambda x: self.format_id(x.split(sep)[0], x.split(sep)[1])
        )

        drkg = drkg.rename(columns={"relation": "relation_type"})
        drkg["resource"] = drkg["relation_type"].apply(lambda x: x.split(sep)[0])
        drkg["pmids"] = ""
        drkg["key_sentence"] = ""

        # Drop columns
        drkg = drkg[
            [
                "source_id",
                "source_type",
                "target_id",
                "target_type",
                "relation_type",
                "resource",
                "key_sentence",
                "pmids",
            ]
        ]

        return [Relation.from_args(**row) for row in drkg.to_dict(orient="records")]  # type: ignore


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

    parser = DrkgParser(
        Path("/Volumes/ProjectData/Current/Datasets/biomedgps/graph_data/entities.tsv"),
        Path("examples"),
        Path("examples"),
    )

    parser.parse()
