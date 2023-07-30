from pathlib import Path
import pandas as pd
import logging
import re
from typing import List
from graph_builder.parsers.base_parser import BaseConfig, BaseParser, Relation, Download

logger = logging.getLogger("graph_builder.parsers.primekg_parser")


class PrimeKGParser(BaseParser):
    """A parser for PrimeKG database. See https://www.nature.com/articles/s41597-023-01960-3 for more details."""

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
            download_url="https://dataverse.harvard.edu/file.xhtml?fileId=6180620&version=2.1#",
            filename="kg.csv",
            is_downloadable=False,
        )

        config = BaseConfig(
            downloads=[download_obj],
            database="primekg",
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

    def format_entity_type(self, raw_type: str) -> str:
        if raw_type == "gene/protein":
            return "Gene"
        elif raw_type == "anatomy":
            return "Anatomy"
        elif raw_type == "biological_process":
            return "BiologicalProcess"
        elif raw_type == "cellular_component":
            return "CellularComponent"
        elif raw_type == "disease":
            return "Disease"
        elif raw_type == "drug":
            return "Compound"
        elif raw_type == "effect/phenotype":
            return "Symptom"
        elif raw_type == "molecular_function":
            return "MolecularFunction"
        elif raw_type == "pathway":
            return "Pathway"
        else:
            return ""

    def format_entity_id(self, raw_id: str, raw_source: str) -> str:
        if raw_source == "NCBI":
            return f"ENTREZ:{raw_id}"
        elif raw_source == "DrugBank":
            return f"DrugBank:{raw_id}"
        elif raw_source == "UBERON":
            return f"UBERON:{raw_id}"
        elif raw_source == "GO":
            return f"GO:{raw_id}"
        elif raw_source == "MONDO":
            return f"MONDO:{raw_id}"
        elif raw_source == "HPO":
            return f"HP:{raw_id}"
        elif raw_source == "REACTOME":
            return f"REACT:{raw_id}"
        else:
            return ""

    def format_relation_type(self, raw_relation_type: str) -> str:
        return raw_relation_type.replace(" ", "_")

    def read_data(self, filepath: Path) -> pd.DataFrame:
        # Specify the column names
        df = pd.read_csv(
            filepath,
            sep=",",
            dtype=str,
        )

        df["source_type"] = df["x_type"].apply(lambda x: self.format_entity_type(x))
        df["target_type"] = df["y_type"].apply(lambda x: self.format_entity_type(x))
        df["source_id"] = df.apply(
            lambda x: self.format_entity_id(x["x_id"], x["x_source"]), axis=1
        )

        df["target_id"] = df.apply(
            lambda x: self.format_entity_id(x["y_id"], x["y_source"]), axis=1
        )

        df["relation_type"] = (
            "PrimeKG::"
            + df["display_relation"]
            + ":"
            + df["source_type"]
            + ":"
            + df["target_type"]
        )

        df["relation_type"] = df["relation_type"].apply(
            lambda x: self.format_relation_type(x)
        )

        df["resource"] = "PrimeKG"
        df["key_sentence"] = ""
        df["pmids"] = ""

        # Remove all unnecessary columns
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
            ]
        ]

        return df

    def extract_relations(self) -> List[Relation]:
        raw_filepath = self.raw_filepaths[0]

        logger.info(f"Read {raw_filepath}")
        primekg = self.read_data(raw_filepath)

        logger.info("Get %d relations" % len(primekg))

        return [Relation.from_args(**row) for row in primekg.to_dict(orient="records")]  # type: ignore


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

    parser = PrimeKGParser(
        Path("/Volumes/ProjectData/Current/Datasets/biomedgps/graph_data/entities.tsv"),
        Path("/Users/jy006/Downloads/Development/biomedgps"),
        Path("/Users/jy006/Downloads/Development/biomedgps"),
    )

    parser.parse()
