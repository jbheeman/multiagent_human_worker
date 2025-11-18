from typing import List, Dict, Optional, Tuple, Literal
from pydantic import BaseModel, Field

class PurchasedProduct(BaseModel):
    """
    A structured model for a single purchased product.
    """
    title: str
    price: str
    asin: str

class UserProducts(BaseModel):
    """
    A model for a user's list of purchased products.
    """
    products: Dict[str, PurchasedProduct]

class ProductKnowledge(BaseModel):
    """
    A structured model to store all gathered information about a single product.
    """
    status: Literal["researching", "pruned"] = Field(
        default="researching",
        description="The current status of this product in the consideration process. 'researching' means it is still a viable option, 'pruned' means it has been eliminated."
    )
    pros: List[str] = Field(
        default_factory=list,
        description="A list of positive attributes or findings about the product."
    )
    cons: List[str] = Field(
        default_factory=list,
        description="A list of negative attributes or findings about the product."
    )
    info: List[str] = Field(
        default_factory=list,
        description="A list of neutral, factual pieces of information about the product."
    )

class KnowledgeBase(BaseModel):
    """
    The central memory of the system, containing all product knowledge and strategic history.
    """
    products: Dict[str, ProductKnowledge] = Field(
        default_factory=dict,
        description="A dictionary where keys are product names and values are their associated knowledge."
    )
    thought_history: List[str] = Field(
        default_factory=list,
        description="A chronological list of strategic thoughts or 'lessons learned' from each refinement loop."
    )

class RefinementDecision(BaseModel):
    """
    A structured model for the PersonaAgent's strategic output after reviewing the KnowledgeBase.
    """
    thought: str = Field(
        description="A high-level summary of the strategic takeaway from this refinement step. This will be used as a memory for future refinement steps to avoid repeating work and to build a coherent strategy."
    )
    next_research_task: str = Field(
        description="The next specific, natural language research task for the ManagerAgent."
    )
    suggested_keywords: List[str] = Field(
        default_factory=list,
        description="A list of optional keywords to help guide the ManagerAgent's next search."
    )
    options_to_prune: List[str] = Field(
        default_factory=list,
        description="A list of product names to remove from consideration."
    )
    status: Literal["continue_refining", "ready_to_choose"] = Field(
        description="Set to 'continue_refining' to loop again, or 'ready_to_choose' to end the search phase."
    )
