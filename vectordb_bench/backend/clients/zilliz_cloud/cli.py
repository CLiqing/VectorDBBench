import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class ZillizTypedDict(CommonTypedDict):
    uri: Annotated[
        str,
        click.option("--uri", type=str, help="uri connection string", required=True),
    ]
    user_name: Annotated[
        str,
        click.option("--user-name", type=str, help="Db username", required=True),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="Zilliz password",
            default=lambda: os.environ.get("ZILLIZ_PASSWORD", ""),
            show_default="$ZILLIZ_PASSWORD",
        ),
    ]
    level: Annotated[
        str,
        click.option("--level", type=str, help="Zilliz index level", required=False),
    ]
    num_shards: Annotated[
        int,
        click.option(
            "--num-shards",
            type=int,
            help="Number of shards",
            required=False,
            default=1,
            show_default=True,
        ),
    ]
    collection_name: Annotated[
        str,
        click.option(
            "--collection-name",
            type=str,
            help="Collection name for Zilliz",
            required=False,
            default="ZillizCloudVDBBench",
            show_default=True,
        ),
    ]
    cardinal_expr_downpush: Annotated[
        bool,
        click.option(
            "--cardinal-expr-downpush",
            is_flag=True,
            help="Enable demo Cardinal expression downpush search params.",
        ),
    ]
    cardinal_expr_threshold: Annotated[
        int | None,
        click.option(
            "--cardinal-expr-threshold",
            type=int,
            required=False,
            default=None,
            help="Optional threshold for id >= threshold downpush validation.",
        ),
    ]
    cardinal_expr_mod_p: Annotated[
        int | None,
        click.option(
            "--cardinal-expr-mod-p",
            type=int,
            required=False,
            default=None,
            help="Optional modulus for id % P < T downpush validation.",
        ),
    ]
    cardinal_expr_mod_t: Annotated[
        int | None,
        click.option(
            "--cardinal-expr-mod-t",
            type=int,
            required=False,
            default=None,
            help="Optional threshold for id % P < T downpush validation.",
        ),
    ]
    cardinal_expr_filtered_out_count: Annotated[
        int | None,
        click.option(
            "--cardinal-expr-filtered-out-count",
            type=int,
            required=False,
            default=None,
            help="Optional filtered-out count for Cardinal downpush strategy.",
        ),
    ]
    cardinal_expr_filter_ratio: Annotated[
        float | None,
        click.option(
            "--cardinal-expr-filter-ratio",
            type=float,
            required=False,
            default=None,
            help="Optional filtered-out ratio for Cardinal downpush strategy.",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(ZillizTypedDict)
def ZillizAutoIndex(**parameters: Unpack[ZillizTypedDict]):
    from .config import AutoIndexConfig, ZillizCloudConfig

    run(
        db=DB.ZillizCloud,
        db_config=ZillizCloudConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
            user=parameters["user_name"],
            password=SecretStr(parameters["password"]),
            num_shards=parameters["num_shards"],
            collection_name=parameters["collection_name"],
        ),
        db_case_config=AutoIndexConfig(
            level=int(parameters["level"]) if parameters["level"] else 1,
            num_shards=parameters["num_shards"],
            cardinal_expr_downpush=parameters["cardinal_expr_downpush"],
            cardinal_expr_threshold=parameters["cardinal_expr_threshold"],
            cardinal_expr_mod_p=parameters["cardinal_expr_mod_p"],
            cardinal_expr_mod_t=parameters["cardinal_expr_mod_t"],
            cardinal_expr_filtered_out_count=parameters[
                "cardinal_expr_filtered_out_count"
            ],
            cardinal_expr_filter_ratio=parameters["cardinal_expr_filter_ratio"],
        ),
        **parameters,
    )
