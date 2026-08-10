import torch
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PdfPipelineOptions,
    LayoutOptions,
)
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_EGRET_LARGE
from docling.datamodel.base_models import InputFormat


def configure_converter():
    accelerator_options = AcceleratorOptions(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    layout_options = LayoutOptions(
        model_spec=DOCLING_LAYOUT_EGRET_LARGE,
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.do_code_enrichment = False
    pipeline_options.do_formula_enrichment = False
    pipeline_options.heading_hierarchy_options.enabled = True
    pipeline_options.generate_parsed_pages = True
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.layout_options = layout_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter


def parse_doc(pdf_path: Path, converter):
    doc = converter.convert(pdf_path)
    return doc.document
