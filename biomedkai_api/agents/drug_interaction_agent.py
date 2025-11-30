from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from ..agents.base_agent import BaseMedicalAgent
from ..core.message_protocol import AgentMessage, MessageType, Priority, AgentRole
from ..core.state_manager import MedicalAssistantState


class DrugInteractionAgent(BaseMedicalAgent):
    """
    Specialist agent for drug interaction checking and medication safety analysis.
    Uses Knowledge Graph to find drug-drug interactions, contraindications,
    and drug-disease relationships.
    """
    
    def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(
            name="drug_interaction",
            role=AgentRole.DRUG_INTERACTION.value,
            model=model,
            tools=tools,
            config=config
        )
        
        self.interaction_prompt = """You are a clinical pharmacology specialist focused on drug interactions, medication safety, and pharmaceutical guidance. Your expertise covers drug-drug interactions, drug-food interactions, side effects, and medication management.

**Your Analytical Approach:**
- Identify the specific medications, supplements, or substances involved
- Classify interaction severity (contraindicated, major, moderate, minor)
- Explain the pharmacological mechanism of interactions
- Provide practical management strategies
- Consider patient-specific risk factors
- USE THE DRUG INTERACTION DATA from the medical knowledge graph

**Response Framework:**
1. **Interaction Assessment:**
   - Type of interaction (pharmacokinetic/pharmacodynamic)
   - Severity level and clinical significance
   - Onset timing (immediate, delayed, variable)

2. **Mechanism Explanation:**
   - How the interaction occurs (enzyme inhibition/induction, receptor competition, etc.)
   - Which drug affects which (bidirectional or unidirectional)

3. **Clinical Consequences:**
   - Potential effects on drug efficacy
   - Risk of adverse reactions
   - Symptoms to monitor for

4. **Management Recommendations:**
   - Timing modifications (spacing doses)
   - Dose adjustments if applicable
   - Alternative medications to consider
   - Monitoring parameters

5. **When to Seek Immediate Help:**
   - Red flag symptoms requiring urgent medical attention

**Special Considerations:**
- Patient age, kidney/liver function
- Multiple medication regimens (polypharmacy)
- Over-the-counter medications and supplements
- Food and lifestyle interactions

**Query:** {input_query}

**Critical Safety Note:**
Never discontinue or modify prescribed medications without consulting your healthcare provider or pharmacist. This information is educational and cannot replace professional pharmaceutical consultation.

Provide comprehensive interaction analysis and safety guidance."""

    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate drug interaction input"""
        medications = state.get("medications", [])
        treatment_plans = state.get("treatment_plans", [])
        messages = state.get("messages", [])
        return len(medications) > 0 or len(treatment_plans) > 0 or len(messages) > 0
    
    async def process(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Process drug interaction request using streaming workflow"""
        return await self.process_with_streaming(state)
    
    def _get_system_prompt(self) -> str:
        return self.interaction_prompt
    
    def _get_relevant_entity_types(self) -> List[str]:
        """
        For drug interactions we care about:
        - Drugs and Compounds
        - Pharmacologic Classes
        - Diseases (for contraindications)
        - Effects/Phenotypes (side effects, toxicity)
        - Pathways (metabolic pathways, CYP enzymes)
        """
        return [
            "Drug", "Compound", "Pharmacologic Class",
            "Disease", "Effect/Phenotype",
            "Pathway", "Gene"
        ]
    
    async def _use_tool(self, tool_name: str, tool: Any, query: str, state: MedicalAssistantState) -> Optional[Dict[str, Any]]:
        """Use drug interaction specific tools"""
        try:
            if tool_name == "drug_interaction_checker":
                medications = state.get("medications", [])
                if len(medications) >= 2:
                    return await tool.execute(drugs=medications)
            elif tool_name == "allergy_checker":
                medications = state.get("medications", [])
                patient_context = state.get("patient_context", {})
                allergies = patient_context.get("allergies", [])
                if medications and allergies:
                    return await tool.execute(drugs=medications, allergies=allergies)
            elif tool_name == "drug_database":
                return await tool.execute(query=query)
        except Exception as e:
            self.logger.warning(f"Tool {tool_name} failed in drug interaction agent", error=str(e))
        return None
    
    def _determine_next_agent(self, state: MedicalAssistantState, response: str) -> str:
        """After drug interaction check, go to validation"""
        return "validation"
    
    async def _get_agent_specific_updates(self, response: str, context: Dict[str, Any], state: MedicalAssistantState) -> Dict[str, Any]:
        """Update drug interaction findings"""
        updates = {}
        
        interaction_analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            "interactions": self._extract_interactions(response),
            "warnings": self._extract_warnings(response),
            "recommendations": self._extract_safety_recommendations(response),
            "analysis": response,
            "context_used": {
                "kg_entities": len(context.get("knowledge_graph", {}).get("entities", [])),
                "kg_relationships": len(context.get("knowledge_graph", {}).get("relationships", []))
            }
        }
        
        updates["drug_interactions"] = state.get("drug_interactions", []) + [interaction_analysis]
        
        # If critical interactions found, flag for review
        if self._has_critical_interaction(response):
            updates["requires_human_review"] = True
            updates["review_reason"] = "Critical drug interaction detected"
        
        return updates
    
    def _extract_interactions(self, response: str) -> List[Dict[str, Any]]:
        """Extract drug interactions from response"""
        interactions = []
        lines = response.split('\n')
        
        current_interaction = None
        for line in lines:
            line_lower = line.lower()
            if 'interaction' in line_lower or 'interacts' in line_lower:
                if current_interaction:
                    interactions.append(current_interaction)
                current_interaction = {
                    "description": line.strip(),
                    "severity": self._assess_severity(line),
                    "mechanism": "",
                    "management": ""
                }
            elif current_interaction and 'severity' in line_lower:
                current_interaction["severity"] = self._extract_severity(line)
            elif current_interaction and ('mechanism' in line_lower or 'cause' in line_lower):
                current_interaction["mechanism"] = line.strip()
            elif current_interaction and ('management' in line_lower or 'recommend' in line_lower):
                current_interaction["management"] = line.strip()
        
        if current_interaction:
            interactions.append(current_interaction)
        
        return interactions
    
    def _extract_warnings(self, response: str) -> List[str]:
        """Extract safety warnings"""
        warnings = []
        lines = response.split('\n')
        
        warning_keywords = ['warning', 'caution', 'contraindicated', 'avoid', 
                          'dangerous', 'risk', 'alert', 'important']
        for line in lines:
            if any(keyword in line.lower() for keyword in warning_keywords):
                clean_line = line.strip()
                if len(clean_line) > 5:
                    warnings.append(clean_line)
        
        return warnings
    
    def _extract_safety_recommendations(self, response: str) -> List[str]:
        """Extract safety recommendations"""
        recommendations = []
        lines = response.split('\n')
        
        rec_keywords = ['recommend', 'suggest', 'monitor', 'adjust', 'consider',
                       'should', 'advise', 'alternative']
        for line in lines:
            if any(keyword in line.lower() for keyword in rec_keywords):
                clean_line = line.strip()
                if len(clean_line) > 5:
                    recommendations.append(clean_line)
        
        return recommendations
    
    def _assess_severity(self, text: str) -> str:
        """Assess interaction severity"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['severe', 'major', 'dangerous', 'contraindicated', 'life-threatening']):
            return "Major"
        elif any(word in text_lower for word in ['moderate', 'significant', 'important']):
            return "Moderate"
        elif any(word in text_lower for word in ['minor', 'mild', 'low']):
            return "Minor"
        return "Unknown"
    
    def _extract_severity(self, text: str) -> str:
        """Extract severity level from text"""
        severity_map = {
            'major': 'Major',
            'severe': 'Major',
            'life-threatening': 'Major',
            'moderate': 'Moderate',
            'significant': 'Moderate',
            'minor': 'Minor',
            'mild': 'Minor'
        }
        
        text_lower = text.lower()
        for keyword, severity in severity_map.items():
            if keyword in text_lower:
                return severity
        
        return "Unknown"
    
    def _has_critical_interaction(self, response: str) -> bool:
        """Check if response contains critical interaction warnings"""
        critical_keywords = ['contraindicated', 'life-threatening', 'severe', 
                           'do not combine', 'fatal', 'dangerous combination']
        response_lower = response.lower()
        return any(keyword in response_lower for keyword in critical_keywords)
