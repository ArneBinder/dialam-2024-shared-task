import dataclasses
from typing import Optional

from pytorch_ie.annotations import BinaryRelation, LabeledSpan, NaryRelation
from pytorch_ie.core import Annotation, AnnotationLayer, annotation_field
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument

# =========================== Annotation Types ============================= #


@dataclasses.dataclass(eq=True, frozen=True)
class Attribute(Annotation):
    annotation: Annotation
    label: str
    type: Optional[str] = None
    score: Optional[float] = dataclasses.field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.label, str):
            raise ValueError("label must be a single string.")
        if not (self.score is None or isinstance(self.score, float)):
            raise ValueError("score must be a single float.")

    def __str__(self) -> str:
        result = f"label={self.label}"
        if self.is_attached:
            result += f",annotation={self.annotation}"
        if self.type is not None:
            result += f",type={self.type}"
        if self.score is not None:
            result += f",score={self.score}"
        return f"{self.__class__.__name__}({result})"


# ============================= Document Types ============================= #


@dataclasses.dataclass
class TextDocumentWithLabeledEntitiesAndEntityAttributes(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    entity_attributes: AnnotationLayer[Attribute] = annotation_field(target="entities")


@dataclasses.dataclass
class TokenDocumentWithLabeledSpans(TokenBasedDocument):
    labeled_spans: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")


@dataclasses.dataclass
class TokenDocumentWithLabeledSpansAndBinaryRelations(TokenDocumentWithLabeledSpans):
    binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(target="labeled_spans")


@dataclasses.dataclass
class TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(
    TokenDocumentWithLabeledSpansAndBinaryRelations
):
    labeled_partitions: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")


@dataclasses.dataclass
class SimplifiedDialAM2024Document(TextBasedDocument):
    l_nodes: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    ya_i2l_nodes: AnnotationLayer[NaryRelation] = annotation_field(target="l_nodes")
    ya_s2ta_nodes: AnnotationLayer[NaryRelation] = annotation_field(target="l_nodes")
    s_nodes: AnnotationLayer[NaryRelation] = annotation_field(target="l_nodes")
