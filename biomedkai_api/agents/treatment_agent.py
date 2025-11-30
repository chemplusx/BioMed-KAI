from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from ..agents.base_agent import BaseMedicalAgent
from ..core.message_protocol import AgentMessage, MessageType, Priority, AgentRole
from ..core.state_manager import MedicalAssistantState


class TreatmentAgent(BaseMedicalAgent):
    """
    Specialist agent for treatment planning and therapy recommendations.
    Uses Knowledge Graph to find drug-disease relationships, contraindications,
    mechanisms of action, and treatment pathways.
    """
    
    def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(
            name="treatment",
            role=AgentRole.TREATMENT.value,
            model=model,
            tools=tools,
            config=config
        )
        
        self.treatment_prompt = """You are a therapeutic guidance specialist providing evidence-based treatment information. Your role is to outline appropriate therapeutic approaches for diagnosed medical conditions, following current clinical guidelines and best practices.

**Your Framework:**
- Base recommendations on current clinical practice guidelines
- Consider patient-specific factors (age, comorbidities, medications)
- Present treatment options in order of typical clinical preference
- Include both pharmacological and non-pharmacological interventions
- Consider contraindications and precautions from the Knowledge Graph
- USE THE DRUG-DISEASE RELATIONSHIPS from the medical knowledge base

**Response Structure for Each Treatment Option:**
1. **Treatment Name/Approach**
2. **Mechanism/Rationale** (how/why it works)
3. **Typical Implementation** (dosing, duration, method)
4. **Expected Outcomes** (what to expect)
5. **Key Considerations** (contraindications, monitoring, side effects)

**Treatment Categories to Consider:**
- First-line therapies (standard of care)
- Adjunctive treatments (supportive care)
- Lifestyle modifications (diet, exercise, behavioral changes)
- Monitoring and follow-up requirements
- Emergency/urgent interventions if applicable

**Patient Context:** {input_query}

**Critical Disclaimer:**
All treatment information is for educational purposes. Actual treatment decisions must always be made by qualified healthcare professionals who can evaluate the complete clinical picture, perform necessary examinations, and consider individual patient factors.

Provide the 5 most appropriate evidence-based therapeutic approaches."""
        
    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate treatment input"""
        diagnosis_history = state.get("diagnosis_history", [])
        conditions = state.get("conditions", [])
        messages = state.get("messages", [])
        return len(diagnosis_history) > 0 or len(conditions) > 0 or len(messages) > 0
    
    async def process(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Process treatment request using streaming workflow"""
        return await self.process_with_streaming(state)
    
    def _get_system_prompt(self) -> str:
        return self.treatment_prompt
    
    def _get_relevant_entity_types(self) -> List[str]:
        """
        For treatment we care about:
        - Drugs/Compounds (treatment options)
        - Diseases (what we're treating)
        - Pharmacologic Classes (drug categories)
        - Pathways (mechanism of action)
        - Effects/Phenotypes (outcomes, side effects)
        """
        return [
            "Drug", "Compound", "Pharmacologic Class",
            "Disease", "Pathway",
            "Effect/Phenotype", "Gene", "Anatomy"
        ]

    async def _use_tool(self, tool_name: str, tool: Any, query: str, state: MedicalAssistantState) -> Optional[Dict[str, Any]]:
        """Use treatment-specific tools"""
        try:
            if tool_name == "guideline_checker":
                # Get the condition from diagnosis
                conditions = state.get("conditions", [])
                if conditions:
                    return await tool.execute(condition=conditions[0], query_type="treatment")
            elif tool_name == "drug_database":
                return await tool.execute(query=query)
            elif tool_name == "pubmed_search":
                return await tool.execute(query=f"treatment {query}")
        except Exception as e:
            self.logger.warning(f"Tool {tool_name} failed in treatment agent", error=str(e))
        return None
    
    def _determine_next_agent(self, state: MedicalAssistantState, response: str) -> str:
        """After treatment, check for drug interactions"""
        # If medications were mentioned, go to drug interaction check
        medications = state.get("medications", [])
        if medications or "medication" in response.lower() or "drug" in response.lower():
            return "drug_interaction"
        return "validation"
    
    async def _get_agent_specific_updates(self, response: str, context: Dict[str, Any], state: MedicalAssistantState) -> Dict[str, Any]:
        """Update treatment plans"""
        updates = {}
        
        treatment_plan = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            "plan": {
                "medications": self._extract_medications(response),
                "non_pharmacological": self._extract_non_pharmacological(response),
                "monitoring": self._extract_monitoring(response),
                "follow_up": self._extract_follow_up(response)
            },
            "rationale": response,
            "context_used": {
                "kg_entities": len(context.get("knowledge_graph", {}).get("entities", [])),
                "kg_relationships": len(context.get("knowledge_graph", {}).get("relationships", []))
            }
        }
        
        updates["treatment_plans"] = state.get("treatment_plans", []) + [treatment_plan]
        
        # Extract medications mentioned to add to state
        meds = self._extract_medications(response)
        if meds:
            current_meds = state.get("medications", [])
            updates["medications"] = list(set(current_meds + meds))
        
        return updates
    
    def _extract_medications(self, response: str) -> List[str]:
        """Extract medication recommendations from response"""
        medications = []
        lines = response.split('\n')
        
        med_keywords = ['medication', 'drug', 'prescribe', 'mg', 'tablet', 'dose', 
                       'first-line', 'therapy', 'treatment']
        
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in med_keywords):
                # Clean up the line
                clean_line = line.strip().lstrip('-*1234567890.').strip()
                if len(clean_line) > 3 and len(clean_line) < 200:
                    medications.append(clean_line)
        
        return medications[:10]  # Limit to 10
    
    def _extract_non_pharmacological(self, response: str) -> List[str]:
        """Extract non-pharmacological recommendations"""
        non_pharm = []
        lines = response.split('\n')
        
        keywords = ['lifestyle', 'exercise', 'diet', 'physical therapy', 'counseling', 
                   'physiotherapy', 'behavioral', 'weight', 'sleep', 'stress']
        for line in lines:
            if any(keyword in line.lower() for keyword in keywords):
                clean_line = line.strip().lstrip('-*1234567890.').strip()
                if len(clean_line) > 3:
                    non_pharm.append(clean_line)
        
        return non_pharm
    
    def _extract_monitoring(self, response: str) -> List[str]:
        """Extract monitoring recommendations"""
        monitoring = []
        lines = response.split('\n')
        
        keywords = ['monitor', 'follow-up', 'check', 'test', 'lab', 'blood work', 
                   'measure', 'assess', 'evaluate']
        for line in lines:
            if any(keyword in line.lower() for keyword in keywords):
                clean_line = line.strip().lstrip('-*1234567890.').strip()
                if len(clean_line) > 3:
                    monitoring.append(clean_line)
        
        return monitoring
    
    def _extract_follow_up(self, response: str) -> List[str]:
        """Extract follow-up recommendations"""
        follow_up = []
        lines = response.split('\n')
        
        keywords = ['follow-up', 'appointment', 'visit', 'weeks', 'months', 
                   'return', 'schedule', 'review']
        for line in lines:
            if any(keyword in line.lower() for keyword in keywords):
                clean_line = line.strip().lstrip('-*1234567890.').strip()
                if len(clean_line) > 3:
                    follow_up.append(clean_line)
        
        return follow_up
