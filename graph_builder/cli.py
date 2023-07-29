import logging
import coloredlogs
import verboselogs
import click
from pathlib import Path
from graph_builder.parsers import parser_map
from joblib import Parallel, delayed


def _parse_database(
    entity_file, db_dir, output_dir, database, download=True, skip=True, num_workers=20
):
    Parser = parser_map.get(database, None)
    if Parser:
        parser = Parser(
            reference_entity_file=entity_file,
            db_directory=db_dir,
            output_directory=output_dir,
            download=download,
            skip=skip,
            num_workers=num_workers
        )
        parsed_results = parser.parse()
    else:
        raise NotSupportedAction("Not supported database: %s" % database)

    return parsed_results


verboselogs.install()
coloredlogs.install(
    fmt="%(asctime)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s"
)
logger = logging.getLogger("graph_builder.cli")


class NotSupportedAction(Exception):
    pass


@click.command(help="Parse databases and make the related graph files.")
@click.option(
    "--db-dir",
    "-d",
    required=True,
    help="The directory which saved the downloaded database files.",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    help="The directory which saved the graph files.",
)
@click.option(
    "--database",
    required=True,
    type=click.Choice(list(parser_map.keys())),
    help="Which databases (you can specify the --database argument multiple times)?",
    multiple=True,
)
@click.option(
    "--ontology-file",
    "-f",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="The ontology file which saved the formatted entities. We will use this file to format the relations in your database.",
)
@click.option("--n-jobs", "-n", required=False, help="Hom many jobs?", default=20)
@click.option(
    "--download/--no-download",
    default=False,
    help="Whether download the source file(s)?",
)
@click.option(
    "--skip/--no-skip", default=True, help="Whether skip the existing file(s)?"
)
def cli(output_dir, db_dir, database, ontology_file, download, n_jobs, skip):
    all_databases = database
    valid_databases = list(
        filter(lambda database: database in parser_map.keys(), all_databases)
    )
    invalid_databases = list(
        filter(lambda database: database not in parser_map.keys(), all_databases)
    )
    if len(invalid_databases) > 0:
        logger.warn(
            "%s databases (%s) is not valid, skip them.",
            len(invalid_databases),
            invalid_databases,
        )
    logger.info(
        "Run jobs with (output_dir: %s, db_dir: %s, databases: %s, download: %s, skip: %s)"
        % (output_dir, db_dir, all_databases, download, skip)
    )

    Parallel(n_jobs=1)(
        delayed(_parse_database)(
            entity_file=Path(ontology_file),
            db_dir=Path(db_dir),
            output_dir=Path(output_dir),
            database=db,
            download=download,
            skip=skip,
            num_workers=n_jobs,
        )
        for db in valid_databases
    )


if __name__ == "__main__":
    cli()
