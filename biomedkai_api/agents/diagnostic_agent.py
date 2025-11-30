from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from ..agents.base_agent import BaseMedicalAgent
from ..core.message_protocol import AgentMessage, MessageType, Priority, AgentRole
from ..core.state_manager import MedicalAssistantState


class DiagnosticAgent(BaseMedicalAgent):
    """
    Specialist agent for medical diagnosis and symptom analysis.
    Uses Knowledge Graph to find disease-symptom relationships,
    drug side effects, and relevant medical pathways.
    """
    
    def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(
            name="diagnostic",
            role=AgentRole.DIAGNOSTIC.value,
            model=model,
            tools=tools,
            config=config
        )
        
        self.diagnostic_prompt = """You are a medical diagnostic reasoning assistant. Your role is to analyze presented symptoms and provide the most likely differential diagnoses based on clinical presentation, considering epidemiology, pathophysiology, and evidence-based medicine.

**Your Approach:**
- Analyze symptoms systematically using clinical reasoning
- Consider patient demographics (age, gender, occupation, medical history)
- Apply diagnostic probability and clinical decision-making principles
- Think through red flags and urgent conditions first
- Consider common conditions before rare ones (common things being common)
- USE THE MEDICAL KNOWLEDGE GRAPH DATA provided to support your reasoning

**Response Format:**
For each of the top 5 most likely diagnoses, provide:
1. **Diagnosis Name**
2. **Probability Assessment** (High/Moderate/Low likelihood)
3. **Supporting Evidence** (2-3 key symptoms/factors that support this diagnosis)
4. **Key Distinguishing Features** (what makes this diagnosis likely given the presentation)

**Clinical Reasoning Process:**
- First identify the primary symptom complex
- Consider anatomical location and system involvement
- Factor in timeline (acute vs chronic)
- Account for patient's medical history and medications
- Apply epidemiological factors (age, gender, occupation)

**Critical Reminder:**
This is for educational/informational purposes only. Always emphasize the need for proper medical evaluation, physical examination, and appropriate diagnostic testing by qualified healthcare professionals.

**Patient Presentation:** {input_query}

Provide your top 5 differential diagnoses with clinical reasoning."""

        self.symptom_analysis_prompt = """Analyze the following symptoms for medical significance:

Symptoms: {symptoms}
Duration: {duration}
Severity: {severity}
Associated Factors: {factors}

Determine:
1. Symptom patterns and clusters
2. Possible organ systems involved
3. Acute vs chronic presentation
4. Severity assessment
5. Need for urgent evaluation
"""
        
    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate diagnostic agent has necessary input"""
        return bool(state.get("symptoms") or 
                   state.get("medical_entities") or
                   (state.get("messages") and len(state["messages"]) > 0))
    
    async def process(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Process diagnostic request using streaming workflow"""
        return await self.process_with_streaming(state)
    
    def _get_system_prompt(self) -> str:
        return self.diagnostic_prompt
    
    def _get_relevant_entity_types(self) -> List[str]:
        """
        For diagnosis we care about:
        - Diseases / conditions
        - Symptoms / phenotypes
        - Drug side effects (could mimic symptoms)
        - Relevant pathways / genes (for mechanistic hints)
        """
        return [
            "Disease", "Symptom", "Effect/Phenotype",
            "Drug", "Compound",
            "Pathway", "Gene", "Anatomy"
        ]

    async def _use_tool(
        self,
        tool_name: str,
        tool: Any,
        query: str,
        state: MedicalAssistantState
    ) -> Optional[Dict[str, Any]]:
        """Use diagnosis-specific tools."""
        try:
            if tool_name == "pubmed_search":
                # Focus the search on diagnosis/differential
                return await tool.execute(
                    query=f"{query} diagnosis differential",
                    max_results=5
                )
            elif tool_name == "guideline_checker":
                # If you have conditions already inferred, use them
                conditions = state.get("conditions", [])
                if conditions:
                    return await tool.execute(
                        condition=conditions[0],
                        query_type="diagnosis"
                    )
            elif tool_name == "symptom_extractor":
                return await tool.execute(text=query)
            else:
                # Generic fallback: pass through the query
                return await tool.execute(query=query)
        except Exception as e:
            self.logger.warning(
                f"Tool {tool_name} failed in diagnostic agent",
                error=str(e)
            )
        return None

    def _determine_next_agent(self, state: MedicalAssistantState, response: str) -> str:
        """After diagnosis, typically go to treatment"""
        if "treatment" in response.lower() or "therapy" in response.lower():
            return "treatment"
        return "validation"
    
    async def _get_agent_specific_updates(self, response: str, context: Dict[str, Any], state: MedicalAssistantState) -> Dict[str, Any]:
        """Update diagnostic findings"""
        updates = {}
        
        # Create diagnostic entry
        diagnostic_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            "diagnosis": {
                "primary_diagnosis": self._extract_primary_diagnosis(response),
                "differential_diagnoses": self._extract_differential_diagnoses(response),
                "confidence": self._calculate_confidence(response)
            },
            "reasoning": response,
            "context_used": {
                "kg_entities": len(context.get("knowledge_graph", {}).get("entities", [])),
                "kg_relationships": len(context.get("knowledge_graph", {}).get("relationships", []))
            }
        }
        
        updates["diagnosis_history"] = state.get("diagnosis_history", []) + [diagnostic_entry]
        
        # Extract conditions mentioned
        conditions = self._extract_conditions(response)
        if conditions:
            updates["conditions"] = list(set(state.get("conditions", []) + conditions))
        
        return updates
    
    def _extract_primary_diagnosis(self, response: str) -> Dict[str, Any]:
        """Extract primary diagnosis from response"""
        lines = response.split('\n')
        for line in lines:
            if 'primary' in line.lower() or 'most likely' in line.lower() or '1.' in line:
                return {
                    "condition": line.strip().lstrip('1.').lstrip('-').strip(),
                    "confidence": self._calculate_confidence(line)
                }
        return {"condition": "See detailed analysis", "confidence": 0.5}
    
    def _extract_differential_diagnoses(self, response: str) -> List[Dict[str, Any]]:
        """Extract differential diagnoses from response"""
        differentials = []
        lines = response.split('\n')
        
        in_differential_section = False
        for line in lines:
            if 'differential' in line.lower() or 'other possibilities' in line.lower():
                in_differential_section = True
                continue
            
            if in_differential_section and (line.strip().startswith('-') or 
                                           line.strip().startswith('2.') or
                                           line.strip().startswith('3.') or
                                           line.strip().startswith('4.') or
                                           line.strip().startswith('5.')):
                condition = line.strip().lstrip('-0123456789.').strip()
                if condition:
                    differentials.append({
                        "condition": condition,
                        "confidence": self._calculate_confidence(line)
                    })
        
        return differentials
    
    def _extract_conditions(self, response: str) -> List[str]:
        """Extract condition names from response"""
        conditions = []
        # Look for common condition patterns
        keywords = ["diagnosis", "condition", "disease", "syndrome", "disorder"]
        lines = response.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                # Simple extraction - could be improved with NLP
                clean_line = line.strip().lstrip('-*1234567890.').strip()
                if len(clean_line) > 3 and len(clean_line) < 100:
                    conditions.append(clean_line)
        
        return conditions[:5]  # Limit to top 5
