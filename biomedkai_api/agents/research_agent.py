from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from ..agents.base_agent import BaseMedicalAgent
from ..core.message_protocol import AgentMessage, MessageType, Priority, AgentRole
from ..core.state_manager import MedicalAssistantState


class ResearchAgent(BaseMedicalAgent):
    """
    Specialist agent for medical research and evidence-based information gathering.
    Uses Knowledge Graph to find gene-disease associations, drug mechanisms,
    biological pathways, and research-relevant relationships.
    """
    
    def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(
            name="research_and_web_search",
            role=AgentRole.RESEARCH.value,
            model=model,
            tools=tools,
            config=config
        )
        
        self.research_prompt = """You are a Medical Research AI Assistant specializing in evidence-based medicine. Your expertise includes:

1. **Literature Review** - Systematic analysis of medical literature
2. **Evidence Synthesis** - Integrate findings from multiple sources
3. **Clinical Trial Analysis** - Evaluate clinical trial data and outcomes
4. **Research Methodology** - Assess study quality and methodology
5. **Emerging Therapies** - Identify cutting-edge treatments and research

**Research Approach:**
- Conduct systematic literature searches
- Evaluate evidence quality and relevance
- Synthesize findings from multiple studies
- USE THE MEDICAL KNOWLEDGE GRAPH for molecular/genetic context
- Assess clinical applicability
- Identify research gaps and limitations
- Provide evidence-based recommendations

**Evidence Standards:**
- Prioritize high-quality studies (RCTs, meta-analyses)
- Consider study limitations and biases
- Evaluate clinical relevance and applicability
- Assess statistical significance and clinical significance
- Consider patient population and setting
- Provide balanced perspective on controversial topics

**Knowledge Graph Integration:**
- Use gene-disease associations for genetic context
- Reference biological pathways for mechanism understanding
- Include drug-target relationships when relevant
- Consider molecular function and cellular components

**Query:** {input_query}

Provide comprehensive, evidence-based research synthesis with attention to the molecular and genetic context from the knowledge graph."""
        
    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate research input"""
        messages = state.get("messages", [])
        return len(messages) > 0
    
    async def process(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Process research request using streaming workflow"""
        return await self.process_with_streaming(state)
    
    def _get_system_prompt(self) -> str:
        return self.research_prompt
    
    def _get_relevant_entity_types(self) -> List[str]:
        """
        For research we want comprehensive molecular/genetic context:
        - Genes and their functions
        - Proteins and molecular functions
        - Biological pathways
        - Diseases and phenotypes
        - Drugs and compounds (for drug discovery)
        """
        return [
            "Gene", "Pathway", "MolecularFunction",
            "BiologicalProcess", "CellularComponent",
            "Disease", "Effect/Phenotype",
            "Drug", "Compound"
        ]
    
    async def _use_tool(self, tool_name: str, tool: Any, query: str, state: MedicalAssistantState) -> Optional[Dict[str, Any]]:
        """Use research-specific tools"""
        try:
            if tool_name == "pubmed_search":
                return await tool.execute(query=query, max_results=10)
            elif tool_name == "clinical_trials_search":
                return await tool.execute(query=query)
            elif tool_name == "web_search":
                return await tool.execute(query=f"medical research {query}")
        except Exception as e:
            self.logger.warning(f"Tool {tool_name} failed in research agent", error=str(e))
        return None
    
    def _determine_next_agent(self, state: MedicalAssistantState, response: str) -> str:
        """After research, go to validation"""
        return "validation"
    
    async def _get_agent_specific_updates(self, response: str, context: Dict[str, Any], state: MedicalAssistantState) -> Dict[str, Any]:
        """Update research findings"""
        updates = {}
        
        research_findings = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            "literature_review": response,
            "key_studies": self._extract_key_studies(response),
            "evidence_level": self._assess_evidence_level(response),
            "clinical_implications": self._extract_clinical_implications(response),
            "context_used": {
                "kg_entities": len(context.get("knowledge_graph", {}).get("entities", [])),
                "kg_relationships": len(context.get("knowledge_graph", {}).get("relationships", []))
            }
        }
        
        updates["research_findings"] = state.get("research_findings", []) + [research_findings]
        
        # Extract genes and pathways mentioned for future context
        kg_context = context.get("knowledge_graph", {})
        entities = kg_context.get("entities", [])
        
        genes = []
        pathways = []
        
        for entity in entities:
            labels = entity.get("labels", [])
            o_label = entity.get("o_label", "")
            name = entity.get("name", "")
            
            if "Gene" in labels or o_label == "Gene":
                genes.append(name)
            elif "Pathway" in labels or o_label == "Pathway":
                pathways.append(name)
        
        if genes:
            updates["genes_mentioned"] = list(set(state.get("genes_mentioned", []) + genes))
        if pathways:
            updates["pathways_mentioned"] = list(set(state.get("pathways_mentioned", []) + pathways))
        
        return updates
    
    def _extract_key_studies(self, response: str) -> List[Dict[str, Any]]:
        """Extract key studies from response"""
        studies = []
        lines = response.split('\n')
        
        current_study = None
        for line in lines:
            line_lower = line.lower()
            if 'study' in line_lower or 'trial' in line_lower or 'research' in line_lower:
                if current_study:
                    studies.append(current_study)
                current_study = {
                    "description": line.strip(),
                    "type": self._identify_study_type(line),
                    "quality": "Unknown"
                }
            elif current_study and any(word in line_lower for word in ['rct', 'randomized', 'meta-analysis', 'systematic']):
                current_study["type"] = self._identify_study_type(line)
                current_study["quality"] = "High"
        
        if current_study:
            studies.append(current_study)
        
        return studies
    
    def _identify_study_type(self, text: str) -> str:
        """Identify study type from text"""
        text_lower = text.lower()
        
        if 'meta-analysis' in text_lower or 'meta analysis' in text_lower:
            return "Meta-analysis"
        elif 'systematic review' in text_lower:
            return "Systematic Review"
        elif 'randomized' in text_lower or 'rct' in text_lower:
            return "Randomized Controlled Trial"
        elif 'cohort' in text_lower:
            return "Cohort Study"
        elif 'case-control' in text_lower:
            return "Case-Control Study"
        elif 'observational' in text_lower:
            return "Observational Study"
        return "Study"
    
    def _assess_evidence_level(self, response: str) -> str:
        """Assess overall evidence level"""
        response_lower = response.lower()
        
        if "systematic review" in response_lower or "meta-analysis" in response_lower:
            return "high"
        elif "randomized controlled trial" in response_lower or "rct" in response_lower:
            return "moderate"
        elif "observational study" in response_lower or "cohort study" in response_lower:
            return "low"
        else:
            return "insufficient"
        
    def _extract_clinical_implications(self, response: str) -> List[str]:
        """Extract clinical implications from response"""
        implications = []
        lines = response.split('\n')
        
        keywords = ['implication', 'recommendation', 'clinical', 'practice', 
                   'application', 'suggest', 'indicate']
        
        for line in lines:
            if any(keyword in line.lower() for keyword in keywords):
                clean_line = line.strip().lstrip('-*1234567890.').strip()
                if len(clean_line) > 5:
                    implications.append(clean_line)
        
        return implications if implications else ["See detailed analysis above"]
