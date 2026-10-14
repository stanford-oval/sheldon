"""What each stage hands back, and the `submit_*` tool generated from it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from sheldon.methods.stage import submit_tool


class PlanOut(BaseModel):
    situation: str = Field(default="", description="The problem and its environment, copied verbatim")
    deliverable: str = Field(default="", description="The specification of the final answer, copied verbatim")
    requirements: list[str] = Field(default_factory=list, description="What the answer must contain, one line each")


class RegionDraft(BaseModel):
    axis: str = Field(default="", description="The dimension this region splits along")
    name: str = Field(default="", description="Short unique name")
    description: str = Field(default="", description="What falls in this region")
    replaces_axis: str = Field(
        default="", description="Only for a new axis that subsumes an existing one: the axis it replaces"
    )


class MergeDraft(BaseModel):
    space: Literal["h", "o"] = Field(default="h", description="h: hypothesis regions; o: observation regions")
    members: list[str] = Field(default_factory=list, description="Two or more existing regions to fold into one")
    name: str = Field(default="", description="The merged region's name: new, or one of the members")
    axis: str = Field(default="", description="The axis it sits on")
    description: str = Field(default="", description="What the merged region covers")
    why: str = Field(default="", description="Why these belong together, or why they go nowhere apart")


class MergeOut(BaseModel):
    merges: list[MergeDraft] = Field(default_factory=list)


class HypothesisDraft(BaseModel):
    statement: str = Field(default="", description="The claim, stated so it can be tested")
    rationale: str = Field(default="", description="Why it could be true, and what would show it false")
    region: str = Field(default="", description="The hypothesis region it sits in")


class ExploreOut(BaseModel):
    hypothesis_regions: list[RegionDraft] = Field(default_factory=list)
    observation_regions: list[RegionDraft] = Field(default_factory=list)
    hypotheses: list[HypothesisDraft] = Field(default_factory=list)
    revive: list[str] = Field(default_factory=list, description="Ids of shelved hypotheses to bring back")


class SelectOut(BaseModel):
    chosen: list[str] = Field(default_factory=list, description="Hypothesis ids, most valuable first")
    reasoning: str = ""


class VerdictDraft(BaseModel):
    hypothesis: str = Field(default="", description="Hypothesis id")
    relation: Literal["supports", "refutes", "neutral"] = "neutral"
    why: str = ""


class ObservationDraft(BaseModel):
    content: str = Field(default="", description="What was found, checkable by another reader")
    region: str = Field(default="", description="The observation region it belongs to")
    sources: list[str] = Field(default_factory=list, description="Retrieval ids it rests on")
    verdicts: list[VerdictDraft] = Field(default_factory=list)


class LocalOut(BaseModel):
    file_path: str = Field(default="", description="Path of the artifact this iteration built, if any")
    statement: str = Field(
        default="",
        description="Only with a revision_basis: the revised claim, stated as a claim (what will happen), "
        "never as a summary of results. Else empty",
    )
    revision_basis: Literal["", "evidence", "approach"] = Field(
        default="",
        description="Why the claim is revised: 'evidence' you gathered contradicts it, or a small change "
        "of 'approach' (not a code fix) should help. Empty when the claim stands",
    )
    rationale: str = Field(default="", description="The evidence or the reasoning behind the revision; empty if none")
    predicts_improvement: bool = Field(
        default=False,
        description="True only if the revised claim predicts a better score than the best so far on this "
        "line. A finding that something does not help is an observation, not a revision",
    )
    notes: str = Field(default="", description="What you built or checked, and what it showed")
    observations: list[ObservationDraft] = Field(default_factory=list)
    done: bool = Field(default=False, description="True when another iteration would not add anything")


class EvidenceOut(BaseModel):
    observations: list[ObservationDraft] = Field(default_factory=list)
    summary: str = ""


class ReviseOp(BaseModel):
    hypothesis: str = Field(default="", description="Hypothesis id")
    op: Literal["keep", "repair", "refine", "shelve"] = "keep"
    statement: str = Field(default="", description="refine only: the updated claim")
    rationale: str = Field(default="", description="Why, citing observation ids")


class ReviseOut(BaseModel):
    ops: list[ReviseOp] = Field(default_factory=list)


class FinalOut(BaseModel):
    answer: str = Field(default="", description="The final answer in exactly the requested format")
    file_path: str = Field(default="", description="If the answer is an artifact on disk, its path")
    confidence: float = 0.5


SUBMIT_MERGES = submit_tool("submit_merges", "Submit the region merges. Call once, at the end.", MergeOut)
SUBMIT_EXPLORATION = submit_tool("submit_exploration", "Submit regions and hypotheses. Call once, at the end.", ExploreOut)
SUBMIT_SELECTION = submit_tool("submit_selection", "Submit the hypotheses to test this round.", SelectOut)
SUBMIT_LOCAL = submit_tool("submit_local", "Submit this iteration's artifact and findings. Call once, at the end.", LocalOut)
SUBMIT_EVIDENCE = submit_tool("submit_evidence", "Submit the observations gathered. Call once, at the end.", EvidenceOut)
SUBMIT_REVISION = submit_tool("submit_revision", "Submit one decision per hypothesis. Call once, at the end.", ReviseOut)
SUBMIT_FINAL = submit_tool("submit_final", "Submit the final answer. Call once, at the end.", FinalOut)

#: every submit tool and its schema, for fixtures that answer by tool name
SUBMIT_SCHEMAS = {
    "submit_merges": MergeOut,
    "submit_exploration": ExploreOut,
    "submit_selection": SelectOut,
    "submit_local": LocalOut,
    "submit_evidence": EvidenceOut,
    "submit_revision": ReviseOut,
    "submit_final": FinalOut,
}
