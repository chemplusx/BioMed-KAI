from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from ..agents.base_agent import BaseMedicalAgent
from ..core.message_protocol import AgentMessage, MessageType, Priority, AgentRole
from ..core.state_manager import MedicalAssistantState


class PreventiveCareAndRiskAssessmentAgent(BaseMedicalAgent):
    """
    Specialist agent for preventive care, health screening, and risk assessment.
    Uses Knowledge Graph to find disease-gene associations, risk factors,
    and preventive interventions.
    """

    def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(
            name="preventive_care_and_risk_assessment",
            role=AgentRole.PREVENTIVE.value,
            model=model,
            tools=tools,
            config=config
        )
        
        self.preventive_prompt = """You are a preventive medicine and health risk assessment specialist. Your focus is on disease prevention, health screening, risk stratification, and health promotion strategies based on current medical guidelines and evidence-based recommendations.

**Your Assessment Framework:**
- Evaluate individual risk factors (age, gender, family history, lifestyle)
- Apply current screening guidelines from authoritative organizations
- Consider cost-effectiveness and benefit-risk ratios
- Provide personalized risk assessment when possible
- USE THE MEDICAL KNOWLEDGE GRAPH to identify disease-gene associations and risk factors
- Address health literacy and patient education needs

**Response Structure:**
1. **Risk Assessment:**
   - Individual risk factors present (from patient context and KG)
   - Risk stratification (low, moderate, high risk)
   - Timeline considerations (immediate, short-term, long-term)

2. **Evidence-Based Recommendations:**
   - Current screening guidelines (USPSTF, specialty societies)
   - Recommended screening intervals
   - Age-appropriate preventive measures

3. **Lifestyle Interventions:**
   - Modifiable risk factors
   - Specific behavioral recommendations
   - Expected impact of interventions

4. **Healthcare Navigation:**
   - When to consult healthcare providers
   - Questions to ask during appointments
   - How to prepare for screenings or tests

5. **Quality and Safety Considerations:**
   - How to evaluate healthcare quality
   - Understanding medical evidence and claims
   - Patient rights and advocacy

**Key Guidelines Sources:**
- U.S. Preventive Services Task Force (USPSTF)
- CDC Prevention Guidelines
- Professional Medical Society Recommendations
- Evidence-based clinical practice guidelines

**Ethical Considerations:**
- Informed consent and shared decision-making
- Cultural sensitivity and health equity
- Patient autonomy and personal values

**Query:** {input_query}

**Educational Purpose:**
This information supports informed healthcare decisions but cannot replace personalized medical consultation. Individual risk assessment and screening decisions should always involve qualified healthcare professionals who can consider your complete medical history and current health status.

Provide comprehensive preventive care guidance and risk assessment."""

        # Preventive care guidelines
        self.guidelines = {
            "cardiovascular": {
                "cholesterol_screening": {"start_age": 35, "interval": 5, "high_risk_interval": 1},
                "blood_pressure": {"start_age": 18, "interval": 2, "high_risk_interval": 1},
                "diabetes_screening": {"start_age": 45, "interval": 3, "high_risk_start": 35}
            },
            "cancer_screening": {
                "mammography": {"start_age": 50, "end_age": 74, "interval": 2, "high_risk_start": 40},
                "cervical": {"start_age": 21, "end_age": 65, "pap_interval": 3, "hpv_interval": 5},
                "colorectal": {"start_age": 50, "end_age": 75, "interval": 10, "high_risk_start": 45},
                "lung_ct": {"start_age": 55, "end_age": 80, "pack_years": 30, "interval": 1}
            },
            "immunizations": {
                "influenza": {"frequency": "annual", "age_start": 6},
                "covid19": {"frequency": "as_recommended", "age_start": 5},
                "pneumococcal": {"age_start": 65, "high_risk_start": 19},
                "shingles": {"age_start": 60, "preferred_age": 60},
                "tdap": {"interval": 10, "pregnancy": True}
            }
        }
        
    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate preventive care has necessary input"""
        patient_context = state.get("patient_context", {})
        messages = state.get("messages", [])
        return bool(patient_context.get("age") or 
                   patient_context.get("medical_history") or
                   state.get("preventive_care_request") or
                   len(messages) > 0)
    
    async def process(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Process preventive care query using streaming workflow"""
        return await self.process_with_streaming(state)
    
    def _get_system_prompt(self) -> str:
        return self.preventive_prompt
    
    def _get_relevant_entity_types(self) -> List[str]:
        """
        For preventive care we care about:
        - Diseases (what we're preventing)
        - Genes (genetic risk factors)
        - Pathways (biological pathways)
        - Effects/Phenotypes (risk factors, outcomes)
        - Drugs/Compounds (preventive medications)
        """
        return [
            "Disease", "Gene", "Pathway",
            "Effect/Phenotype", "Drug", "Compound",
            "BiologicalProcess", "Anatomy"
        ]

    async def _use_tool(self, tool_name: str, tool: Any, query: str, state: MedicalAssistantState) -> Optional[Dict[str, Any]]:
        """Use preventive care specific tools"""
        try:
            if tool_name == "cv_risk_calculator":
                patient_context = state.get("patient_context", {})
                return await tool.execute(**patient_context)
            elif tool_name == "guideline_checker":
                conditions = state.get("conditions", [])
                if conditions:
                    return await tool.execute(condition=conditions[0], query_type="screening")
            elif tool_name == "pubmed_search":
                return await tool.execute(query=f"prevention screening {query}")
        except Exception as e:
            self.logger.warning(f"Tool {tool_name} failed in preventive care agent", error=str(e))
        return None
    
    def _determine_next_agent(self, state: MedicalAssistantState, response: str) -> str:
        """Determine next agent based on risk assessment"""
        response_lower = response.lower()
        
        # High risk findings should go to diagnostic
        if any(word in response_lower for word in ['high risk', 'elevated risk', 'significant risk']):
            return "diagnostic"
        
        # Normal preventive care goes to validation
        return "validation"
    
    async def _get_agent_specific_updates(self, response: str, context: Dict[str, Any], state: MedicalAssistantState) -> Dict[str, Any]:
        """Update state with preventive care findings"""
        updates = {}
        
        # Extract risk level from response
        risk_level = self._extract_risk_level(response)
        
        preventive_assessment = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            "risk_level": risk_level,
            "recommendations": self._extract_recommendations(response),
            "screenings_due": self._extract_screenings(response),
            "context_used": {
                "kg_entities": len(context.get("knowledge_graph", {}).get("entities", [])),
                "kg_relationships": len(context.get("knowledge_graph", {}).get("relationships", []))
            }
        }
        
        updates["preventive_assessment"] = preventive_assessment
        updates["risk_level"] = risk_level
        
        # Extract any conditions mentioned
        kg_context = context.get("knowledge_graph", {})
        entities = kg_context.get("entities", [])
        
        conditions = []
        for entity in entities:
            labels = entity.get("labels", [])
            o_label = entity.get("o_label", "")
            name = entity.get("name", "")
            
            if "Disease" in labels or o_label == "Disease":
                conditions.append(name)
        
        if conditions:
            updates["conditions"] = list(set(state.get("conditions", []) + conditions))
        
        return updates
    
    def _extract_risk_level(self, response: str) -> str:
        """Extract risk level from response"""
        response_lower = response.lower()
        
        if any(word in response_lower for word in ['high risk', 'high-risk', 'elevated risk']):
            return "high"
        elif any(word in response_lower for word in ['moderate risk', 'moderate-risk', 'intermediate']):
            return "moderate"
        elif any(word in response_lower for word in ['low risk', 'low-risk', 'minimal']):
            return "low"
        return "moderate"  # Default
    
    def _extract_recommendations(self, response: str) -> List[str]:
        """Extract recommendations from response"""
        recommendations = []
        lines = response.split('\n')
        
        rec_keywords = ['recommend', 'suggest', 'advise', 'should', 'consider',
                       'screening', 'lifestyle', 'intervention']
        
        for line in lines:
            if any(keyword in line.lower() for keyword in rec_keywords):
                clean_line = line.strip().lstrip('-*1234567890.').strip()
                if len(clean_line) > 5:
                    recommendations.append(clean_line)
        
        return recommendations
    
    def _extract_screenings(self, response: str) -> List[str]:
        """Extract screening recommendations from response"""
        screenings = []
        lines = response.split('\n')
        
        screening_keywords = ['screening', 'test', 'mammogram', 'colonoscopy', 
                            'blood pressure', 'cholesterol', 'glucose', 'pap']
        
        for line in lines:
            if any(keyword in line.lower() for keyword in screening_keywords):
                clean_line = line.strip().lstrip('-*1234567890.').strip()
                if len(clean_line) > 5:
                    screenings.append(clean_line)
        
        return screenings
