from pathlib import Path
import polars as pl
import pandas as pd
import csv
import gzip
import logging
from typing import List
from graph_builder.parsers.base_parser import BaseConfig, BaseParser, Relation, Download

logger = logging.getLogger("graph_builder.parsers.ctd_parser")


class BaseExtractor:
    def __init__(self, basedir: Path, output_dir: Path) -> None:
        self.database_name = "CTD"
        self.basedir = basedir
        self.output_dir = output_dir

    def _load_data(self, filepath: Path) -> pl.DataFrame:
        """Load data from a csv file."""
        if self._check_file_exists(filepath) is False:
            raise ValueError(f"File {filepath} not found.")

        self.headers = self._find_fields(filepath)
        logging.info("Found headers %s in %s" % (self.headers, filepath))
        data = self._read_data(filepath, self.headers)
        logging.info("Read %s rows from %s." % (data.shape[0], filepath))
        return data

    def _check_file_exists(self, filepath: Path):
        """Check if a file exists."""
        return filepath.exists()

    def write(self, df, output_filename: str, sep="\t"):
        """Write the extracted entities to a csv file."""
        outputpath = self.output_dir / output_filename
        df.write_csv(outputpath, separator=sep)
        logging.info("Done! The extracted entities are saved in %s.\n" % outputpath)

    def _read_data(self, filepath: Path, headers=[]):
        """Skip rows that start with '#' in a csv file."""
        df = pl.read_csv(
            filepath,
            has_header=False,
            infer_schema_length=0,
            new_columns=headers,
            comment_char="#",
            separator="\t",
        )

        return df

    def _find_fields(self, filepath):
        """Find the columns that contain a certain field."""
        with gzip.open(filepath, mode="rt") as f:
            reader = csv.reader(f, delimiter="\t")  # type: ignore

            fields = []
            for line in reader:
                content = line[0]
                if content.startswith("# Fields:"):
                    header = next(reader)
                    fields = [field.strip("# ") for field in header]
                    break

            return fields


class RelationshipExtractor(BaseExtractor):
    def __init__(
        self,
        filepath: Path,
        relation_type,
        db_dir: Path = Path("."),
        output_dir: Path = Path("."),
    ) -> None:
        super().__init__(db_dir, output_dir)

        self.relation_label = "IS_ASSOCIATED_WITH"
        self.relation_type = relation_type
        self.filepath = filepath
        self.extract()

    def extract(self) -> pl.DataFrame:
        """Extract relationships from the data."""
        # TODO: Add dtype for some columns, now it's string by default.
        data = self._load_data(self.filepath)

        relation_type = self.relation_type.lower()
        method_name = f"_extract_{relation_type}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(data)
        else:
            raise ValueError(f"Relationship type {relation_type} is not supported.")

    def _extract_relationship(
        self,
        df,
        default_extracted_columns,
        renamed_columns,
        label,
        groupby_columns=[],
        merge_funcs={},
    ) -> pl.DataFrame:
        for column in default_extracted_columns:
            if column not in df.columns:
                raise ValueError(f"Column {column} not found in the dataframe.")

        df = df.select(default_extracted_columns)
        df = df.rename(
            {old: new for old, new in zip(default_extracted_columns, renamed_columns)}
        )

        source_type = self.relation_type.split("_")[0]
        target_type = self.relation_type.split("_")[1]
        new_df = pl.DataFrame(
            {
                "relation_type": ["%s:%s:%s" % (label, source_type, target_type)]
                * df.shape[0],
                "source_type": [source_type] * df.shape[0],
                "target_type": [target_type] * df.shape[0],
                "resource": [self.database_name] * df.shape[0],
                "key_sentence": [""] * df.shape[0],
            }
        )
        df = df.hstack(new_df)

        logging.info(
            "Extract %s relationships from %s..." % (df.shape[0], self.filepath)
        )
        logging.info(
            "Keep %s columns: %s"
            % (len(default_extracted_columns), default_extracted_columns)
        )
        logging.info("Rename columns to: %s" % renamed_columns)

        if groupby_columns and merge_funcs:
            # Add default columns which are not in groupby_columns but defined in current function.
            if sorted(groupby_columns) == sorted(["source_id", "target_id"]):
                groupby_columns = [
                    "source_id",
                    "target_id",
                    "relation_type",
                    "source_type",
                    "target_type",
                    "resource",
                ]
            df = df.groupby(groupby_columns).agg(merge_funcs)
            logging.info("After groupby, %s relationships are left." % df.shape[0])
        else:
            df = df.unique()
            logging.info(
                "Drop duplicates in place, and %s relationships are left." % df.shape[0]
            )

        return df

    def _extract_compound_gene(self, df) -> pl.DataFrame:
        """Extract compound-gene relationships from CTD_chem_gene_ixns.csv"""
        default_extracted_columns = ["ChemicalID", "GeneID"‘]
        renamed_columns = ["source_id", "target_id"]
        label = "CTD::IS_ASSOCIATED_WITH"
        df = df.with_columns(pl.col("GeneID").apply(lambda x: "ENTREZ:" + x))
        df = df.with_columns(pl.col("ChemicalID").apply(lambda x: "MESH:" + x))

        return self._extract_relationship(
            df, default_extracted_columns, renamed_columns, label
        )

    def _extract_compound_disease(self, df) -> pl.DataFrame:
        """Extract compound-disease relationships from CTD_chemicals_diseases.csv"""
        default_extracted_columns = ["ChemicalID", "DiseaseID"]
        renamed_columns = ["source_id", "target_id"]
        df = df.with_columns(pl.col("ChemicalID").apply(lambda x: "MESH:" + x))
        label = "CTD::IS_ASSOCIATED_WITH"

        return self._extract_relationship(
            df, default_extracted_columns, renamed_columns, label
        )

    def _extract_compound_pathway(self, df) -> pl.DataFrame:
        """Extract compound-pathway relationships from CTD_chem_pathways_enriched.csv"""
        default_extracted_columns = ["ChemicalID", "PathwayID"]
        renamed_columns = ["source_id", "target_id"]
        df = df.with_columns(pl.col("ChemicalID").apply(lambda x: "MESH:" + x))
        label = "CTD::IS_ASSOCIATED_WITH"

        return self._extract_relationship(
            df, default_extracted_columns, renamed_columns, label
        )

    def _extract_gene_disease(self, df) -> pl.DataFrame:
        """Extract gene-disease relationships from CTD_genes_diseases.csv"""
        default_extracted_columns = [
            "GeneID",
            "DiseaseID",
            "DirectEvidence",
            "InferenceScore",
            "InferenceChemicalName",
            "PubMedIDs",
        ]
        renamed_columns = [
            "source_id",
            "target_id",
            "evidence",
            "degree",
            "induced_by",
            "pmids",
        ]
        label = "CTD::IS_ASSOCIATED_WITH"
        df = df.with_columns(pl.col("GeneID").apply(lambda x: "ENTREZ:" + x))
        df = df.with_columns(pl.col("InferenceScore").cast(pl.Float64))

        merge_funcs = [
            pl.col("evidence")
            .filter(pl.col("evidence") != "")
            .unique()
            .str.concat("|"),

            pl.col("degree").max(),
            
            pl.col("induced_by")
            .filter(pl.col("induced_by") != "")
            .unique()
            .str.concat("|"),
            
            # TODO: Any deduplicated pmids exist after unique()?
            pl.col("pmids").filter(pl.col("pmids") != "").unique().str.concat("|"),
        ]

        return self._extract_relationship(
            df,
            default_extracted_columns,
            renamed_columns,
            label,
            groupby_columns=["source_id", "target_id"],
            merge_funcs=merge_funcs,
        )

    def _extract_gene_pathway(self, df) -> pl.DataFrame:
        """Extract gene-pathway relationships from CTD_genes_pathways.csv"""
        default_extracted_columns = ["GeneID", "PathwayID"]
        renamed_columns = ["source_id", "target_id"]
        label = "CTD::IS_ASSOCIATED_WITH"
        df = df.with_columns(pl.col("GeneID").apply(lambda x: "ENTREZ:" + x))

        return self._extract_relationship(
            df, default_extracted_columns, renamed_columns, label
        )

    def _extract_disease_biologicalprocess(self, df) -> pl.DataFrame:
        """Extract disease-biological process relationships from CTD_Phenotype-Disease_biological_process_associations.csv"""
        default_extracted_columns = ["DiseaseID", "GOID"]
        renamed_columns = ["source_id", "target_id"]
        label = "CTD::IS_ASSOCIATED_WITH"

        return self._extract_relationship(
            df, default_extracted_columns, renamed_columns, label
        )

    def _extract_disease_cellularcomponent(self, df) -> pl.DataFrame:
        """Extract disease-cellular component relationships from CTD_Phenotype-Disease_cellular_component_associations.csv"""
        default_extracted_columns = ["DiseaseID", "GOID"]
        renamed_columns = ["source_id", "target_id"]
        label = "CTD::IS_ASSOCIATED_WITH"

        return self._extract_relationship(
            df, default_extracted_columns, renamed_columns, label
        )

    def _extract_disease_molecularfunction(self, df) -> pl.DataFrame:
        """Extract disease-molecular function relationships from CTD_Phenotype-Disease_molecular_function_associations.csv"""
        default_extracted_columns = ["DiseaseID", "GOID"]
        renamed_columns = ["source_id", "target_id"]
        label = "CTD::IS_ASSOCIATED_WITH"

        return self._extract_relationship(
            df, default_extracted_columns, renamed_columns, label
        )

    def _extract_disease_pathway(self, df) -> pl.DataFrame:
        """Extract disease-pathway relationships from CTD_diseases_pathways.csv"""
        default_extracted_columns = ["DiseaseID", "PathwayID"]
        renamed_columns = ["source_id", "target_id"]
        label = "CTD::IS_ASSOCIATED_WITH"

        return self._extract_relationship(
            df, default_extracted_columns, renamed_columns, label
        )


class CtdParser(BaseParser):
    def __init__(
        self,
        reference_entity_file: Path,
        db_directory: Path,
        output_directory: Path,
        download=True,
        skip=True,
        num_workers: int = 20,
    ):
        chem_gene = Download(
            download_url="http://ctdbase.org/reports/CTD_chem_gene_ixns.tsv.gz",
            filename="CTD_chem_gene_ixns.tsv.gz",
        )

        chem_disease = Download(
            download_url="http://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz",
            filename="CTD_chemicals_diseases.tsv.gz",
        )

        chem_pathway = Download(
            download_url="http://ctdbase.org/reports/CTD_chem_pathways_enriched.tsv.gz",
            filename="CTD_chem_pathways_enriched.tsv.gz",
        )

        gene_disease = Download(
            download_url="http://ctdbase.org/reports/CTD_genes_diseases.tsv.gz",
            filename="CTD_genes_diseases.tsv.gz",
        )

        gene_pathway = Download(
            download_url="http://ctdbase.org/reports/CTD_genes_pathways.tsv.gz",
            filename="CTD_genes_pathways.tsv.gz",
        )

        disease_bp = Download(
            download_url="http://ctdbase.org/reports/CTD_Phenotype-Disease_biological_process_associations.tsv.gz",
            filename="CTD_Phenotype-Disease_biological_process_associations.tsv.gz",
        )

        disease_cc = Download(
            download_url="http://ctdbase.org/reports/CTD_Phenotype-Disease_cellular_component_associations.tsv.gz",
            filename="CTD_Phenotype-Disease_cellular_component_associations.tsv.gz",
        )

        disease_mf = Download(
            download_url="http://ctdbase.org/reports/CTD_Phenotype-Disease_molecular_function_associations.tsv.gz",
            filename="CTD_Phenotype-Disease_molecular_function_associations.tsv.gz",
        )

        disease_pathway = Download(
            download_url="http://ctdbase.org/reports/CTD_diseases_pathways.tsv.gz",
            filename="CTD_diseases_pathways.tsv.gz",
        )

        self.default_relationships = {
            "Compound_Gene": chem_gene.filename,
            "Compound_Disease": chem_disease.filename,
            "Compound_Pathway": chem_pathway.filename,
            "Gene_Disease": gene_disease.filename,
            "Gene_Pathway": gene_pathway.filename,
            "Disease_BiologicalProcess": disease_bp.filename,
            "Disease_CellularComponent": disease_cc.filename,
            "Disease_MolecularFunction": disease_mf.filename,
            "Disease_Pathway": disease_pathway.filename,
        }

        config = BaseConfig(
            downloads=[
                chem_gene,
                chem_disease,
                chem_pathway,
                gene_disease,
                gene_pathway,
                disease_bp,
                disease_cc,
                disease_mf,
                disease_pathway,
            ],
            database="ctd",
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

    def extract_relations(self) -> List[Relation]:
        data: List[pd.DataFrame] = []
        for filepath in self.raw_filepaths:
            filename = filepath.name.split("/")[-1]
            filepath_relations = {v: k for k, v in self.default_relationships.items()}
            relation_type = filepath_relations.get(filename, None)
            relation_extractor = RelationshipExtractor(
                filepath, relation_type, self.db_directory, self.output_directory
            )

            df = relation_extractor.extract()
            df = pl.DataFrame.to_pandas(df)
            data.append(df)

        df = pd.concat(data, axis=0, ignore_index=True)
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

        return [Relation.from_args(**row) for row in df.to_dict(orient="records")]  # type: ignore


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

    parser = CtdParser(
        Path("/Volumes/ProjectData/Current/Datasets/biomedgps/graph_data/entities.tsv"),
        Path("/Users/jy006/Downloads/Development/biomedgps/ctd"),
        Path("/Users/jy006/Downloads/Development/biomedgps_output/ctd"),
    )

    parser.parse()
