from typing import Dict, Type
from graph_builder.parsers.base_parser import BaseParser
from graph_builder.parsers.drkg_parser import DrkgParser

parser_map: Dict[str, Type[BaseParser]] = {
    "drkg": DrkgParser,
}