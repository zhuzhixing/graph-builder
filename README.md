# Graph Builder

For building a knowledge graph, we need to integrate different public databases. In this repository, we provide a framework to build a knowledge graph with different public databases. If you want to integrate a new database, you only need to write a new parser for the database. The framework will automatically parse the new database and generate a file which contains formatted relations. Then, you can use all formatted relations and related entities to build a knowledge graph.

## Installation

```bash
mkdir graph_database
cd graph_database

virtualenv -p python3 .env
source .env/bin/activate

pip install git+https://github.com/yjcyxky/graph-builder.git
```

## Usage

```bash
(.env) ➜  graph-builder git:(main) ✗ graph-builder --help
Usage: graph-builder [OPTIONS]

  Parse databases and make the related graph files.

Options:
  -d, --db-dir TEXT               The directory which saved the downloaded
                                  database files.  [required]
  -o, --output-dir TEXT           The directory which saved the graph files.
                                  [required]
  --database [drkg|ctd|hsdn|primekg]
                                  Which databases (you can specify the
                                  --database argument multiple times)?
                                  [required]
  -f, --ontology-file FILE        The ontology file which saved the formatted
                                  entities. We will use this file to format
                                  the relations in your database.  [required]
  -n, --n-jobs INTEGER            Hom many jobs?
  --download / --no-download      Whether download the source file(s)?
  --skip / --no-skip              Whether skip the existing file(s)?
  -l, --log-file TEXT             The log file.
  --debug / --no-debug            Whether enable the debug mode?
  --help                          Show this message and exit.
```

## Example

Download the HDSN database and build the graph files. We assume that you want to save the results in the `~/Downloads/Development/biomedgps_output/hsdn` directory, and downloaded files into the `~/Downloads/Development/biomedgps/hsdn` directory. So you can run the following command:

In the command, you need to specify a entity file which contains the formatted entities. We will use this file to format the relations in your database. You can build the entity file by using the [ontology-matcher](https://github.com/yjcyxky/ontology-matcher) and [biomedgps-data](https://github.com/yjcyxky/biomedgps-data) repositories. In the most cases, you don't need to build the entity file by yourself, so you can download the entity file from the [biomedgps-data]() repository.

After ran the command, you can find the log file in the `~/Downloads/Development/biomedgps_output/hsdn/log.txt` file and the graph files in the `~/Downloads/Development/biomedgps_output/hsdn` directory. You may get four tsv files: duplicated_hsdn.tsv, formatted_hsdn.tsv, hsdn.entity_id_map.json, invalid_hsdn.tsv. The `formatted_hsdn.tsv` file contains the formatted relations, and the `hsdn.entity_id_map.json` file contains the mapping between the original entity id and the formatted entity id. The `invalid_hsdn.tsv` file contains the invalid relations which cannot be formatted by the entity file. The `duplicated_hsdn.tsv` file contains the duplicated relations which have the same relation type, source id, target id, and pmid.

```bash
graph-builder --database hsdn -d ~/Downloads/Development/biomedgps -o ~/Downloads/Development/biomedgps_output -f /Volumes/ProjectData/Current/Datasets/biomedgps/graph_data/entities.tsv -n 20 --download --skip -l ~/Downloads/Development/biomedgps_output/hsdn/log.txt --debug
```
