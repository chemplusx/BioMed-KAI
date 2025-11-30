from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from ..agents.base_agent import BaseMedicalAgent
from ..core.message_protocol import AgentMessage, MessageType, Priority, AgentRole
from ..core.state_manager import MedicalAssistantState


class GeneralMedicalAgent(BaseMedicalAgent):
    """
    Specialist agent for general medical education and health information.
    Handles questions about anatomy, physiology, general health concepts,
    and medical terminology using the Knowledge Graph for context.
    """
    
    def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(
            name="general_medical_query",
            role=AgentRole.GENERAL.value,
            model=model,
            tools=tools,
            config=config
        )
        
        self.general_prompt = """You are a knowledgeable medical education assistant specializing in general health information. Your role is to provide accurate, evidence-based explanations about common health topics, bodily functions, and general wellness concepts.

**Your Guidelines:**
- Provide clear, educational explanations using accessible language
- Include relevant medical terminology with simple definitions
- Use analogies and examples to make complex concepts understandable
- Focus on established medical knowledge and consensus
- USE THE MEDICAL KNOWLEDGE GRAPH DATA to provide context and relationships
- Always emphasize that this is educational information, not personalized medical advice

**Response Structure:**
1. Direct answer to the question
2. Brief explanation of underlying mechanisms when relevant
3. Related medical entities and their connections (from Knowledge Graph)
4. Additional context or related information that might be helpful
5. When appropriate, mention when to consult healthcare professionals

**Important Reminders:**
- Do not provide specific medical advice or diagnose conditions
- Encourage consultation with healthcare providers for personal health concerns
- Present information objectively without unnecessary alarm

**Query:** {input_query}

Provide a comprehensive yet accessible explanation."""

        # Emergency criteria thresholds for triage if needed
        self.emergency_criteria = {
            "vital_signs": {
                "hr_critical": {"min": 40, "max": 150},
                "bp_critical": {"systolic_min": 70, "systolic_max": 200, "diastolic_max": 110},
                "rr_critical": {"min": 8, "max": 30},
                "temp_critical": {"min": 94, "max": 104},
                "o2_sat_critical": {"min": 85}
            },
            "red_flag_symptoms": [
                "chest pain", "difficulty breathing", "severe headache",
                "altered mental status", "severe abdominal pain",
                "uncontrolled bleeding", "stroke symptoms",
                "severe allergic reaction", "poisoning"
            ]
        }
        
    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate general medical has necessary input"""
        return bool(state.get("symptoms") or 
                   state.get("chief_complaint") or
                   (state.get("messages") and len(state["messages"]) > 0))
    
    async def process(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Process general medical query using streaming workflow"""
        return await self.process_with_streaming(state)
    
    def _get_system_prompt(self) -> str:
        return self.general_prompt
    
    def _get_relevant_entity_types(self) -> List[str]:
        """
        For general medical queries we want broad coverage:
        - Anatomy (body structures)
        - Biological processes
        - Diseases
        - Symptoms/Phenotypes
        - Drugs/Compounds
        - Pathways
        """
        return [
            "Anatomy", "BiologicalProcess", "CellularComponent",
            "Disease", "Symptom", "Effect/Phenotype",
            "Drug", "Compound", "Pathway", "Gene"
        ]

    async def _use_tool(self, tool_name: str, tool: Any, query: str, state: MedicalAssistantState) -> Optional[Dict[str, Any]]:
        """Use general medical tools"""
        try:
            if tool_name == "pubmed_search":
                return await tool.execute(query=query, max_results=5)
            elif tool_name == "symptom_analyzer":
                symptoms = state.get("symptoms", [])
                if symptoms:
                    return await tool.execute(symptoms=symptoms, mode="education")
        except Exception as e:
            self.logger.warning(f"Tool {tool_name} failed in general medical agent", error=str(e))
        return None
    
    def _determine_next_agent(self, state: MedicalAssistantState, response: str) -> str:
        """Determine next agent based on response content"""
        response_lower = response.lower()
        
        # Check for emergency indicators
        if self._check_emergency_indicators(response_lower):
            return "validation"  # Immediate validation for emergencies
        
        # Check if response suggests diagnosis
        if any(word in response_lower for word in ['diagnose', 'diagnosis', 'condition', 'disease']):
            return "diagnostic"
        
        # Check if response suggests treatment
        if any(word in response_lower for word in ['treatment', 'therapy', 'medication']):
            return "treatment"
        
        # General information usually ends the flow
        return "validation"
    
    def _check_emergency_indicators(self, text: str) -> bool:
        """Check for emergency indicators in text"""
        emergency_keywords = [
            "emergency", "urgent", "critical", "severe",
            "chest pain", "difficulty breathing", "unconscious"
        ]
        return any(keyword in text for keyword in emergency_keywords)
    
    async def _get_agent_specific_updates(self, response: str, context: Dict[str, Any], state: MedicalAssistantState) -> Dict[str, Any]:
        """Update state with general medical information"""
        updates = {}
        
        # Check for any conditions or symptoms mentioned in the context
        kg_context = context.get("knowledge_graph", {})
        entities = kg_context.get("entities", [])
        
        symptoms = []
        conditions = []
        
        for entity in entities:
            labels = entity.get("labels", [])
            o_label = entity.get("o_label", "")
            name = entity.get("name", "")
            
            if "Symptom" in labels or o_label == "Symptom":
                symptoms.append(name)
            elif "Disease" in labels or o_label == "Disease":
                conditions.append(name)
        
        if symptoms:
            updates["symptoms"] = list(set(state.get("symptoms", []) + symptoms))
        if conditions:
            updates["conditions"] = list(set(state.get("conditions", []) + conditions))
        
        # Check for emergency flags in response
        if self._check_emergency_indicators(response.lower()):
            updates["emergency_flag"] = True
        
        return updates
