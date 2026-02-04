"""
Failure Analysis - Analyze Partial and Failed KG Segments
==========================================================
Reads validation report and sends failed/partial segments to LLM for analysis
"""

import json
import logging
from typing import Dict, List, Any
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class FailureAnalyzer:
    """Analyze why segments failed or were partial"""
    
    def __init__(self, api_key: str, model: str, api_base: str):
        self.llm = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
    
    def analyze_segment_failure(self, segment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single failed/partial segment using LLM
        
        Args:
            segment_data: Full segment validation data
        
        Returns:
            Analysis result with explanation and recommendations
        """
        seg_id = segment_data.get("segment_id", "unknown")
        status = segment_data.get("validation", {}).get("status", "UNKNOWN")
        entity_score = segment_data.get("validation", {}).get("entity_score", 0.0)
        rel_score = segment_data.get("validation", {}).get("relationship_score", 0.0)
        overall_score = segment_data.get("validation", {}).get("overall_score", 0.0)
        
        # Collect relationship failures
        failed_relationships = []
        for rel in segment_data.get("relationships_validation", []):
            if rel["score"] < 0.7:
                failed_relationships.append({
                    "relationship": rel["relationship"],
                    "score": rel["score"],
                    "feedback": rel.get("feedback", "")
                })
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(
            seg_id, status, entity_score, rel_score, overall_score,
            segment_data.get("entity_names", []),
            failed_relationships,
            segment_data.get("segment_text", "")
        )
        
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                timeout=30
            )
            
            analysis_text = response.choices[0].message.content
            
            return {
                "segment_id": seg_id,
                "status": status,
                "scores": {
                    "entity": entity_score,
                    "relationship": rel_score,
                    "overall": overall_score
                },
                "analysis": analysis_text,
                "failed_relationships_count": len(failed_relationships)
            }
        
        except Exception as e:
            logger.error(f"Failed to analyze segment {seg_id}: {e}")
            return {
                "segment_id": seg_id,
                "status": status,
                "scores": {
                    "entity": entity_score,
                    "relationship": rel_score,
                    "overall": overall_score
                },
                "analysis": f"Error during analysis: {str(e)}",
                "failed_relationships_count": len(failed_relationships)
            }
    
    def _build_analysis_prompt(
        self,
        seg_id: str,
        status: str,
        entity_score: float,
        rel_score: float,
        overall_score: float,
        entity_names: List[str],
        failed_relationships: List[Dict],
        segment_text: str
    ) -> str:
        """Build prompt for LLM failure analysis"""
        
        rel_details = "\n".join([
            f"  - {r['relationship']} (Score: {r['score']:.2f}): {r['feedback']}"
            for r in failed_relationships
        ])
        
        return f"""You are a Knowledge Graph quality analyst. Analyze why this segment validation failed or was only partial.

**SEGMENT ID:** {seg_id}
**STATUS:** {status}
**SCORES:**
- Entity Score: {entity_score:.2f}
- Relationship Score: {rel_score:.2f}
- Overall Score: {overall_score:.2f}

**SEGMENT TEXT:**
{segment_text[:500]}{"..." if len(segment_text) > 500 else ""}

**ENTITIES EXTRACTED:**
{", ".join(entity_names)}

**FAILED/LOW-SCORE RELATIONSHIPS:**
{rel_details if failed_relationships else "None - all relationships passed"}

**ANALYSIS TASK:**
1. Identify the primary reason(s) for failure or partial validation
2. Determine if the issue is:
   - Missing evidence in PDF
   - Incorrect entity extraction
   - Hallucinated relationships
   - Ambiguous relationship types
   - Other data quality issues
3. Provide specific, actionable recommendations to improve this segment

**OUTPUT FORMAT:**
Provide a concise analysis (3-5 sentences) focusing on:
- Root cause of low score
- Specific problematic relationships or entities
- Recommended fixes

Keep your response direct and actionable."""
    
    def analyze_report(self, report_path: str, output_path: str = "failure_analysis_report.json"):
        """
        Analyze entire validation report and generate failure analysis
        
        Args:
            report_path: Path to validation report JSON
            output_path: Path to save analysis report
        """
        logger.info(f"Loading validation report from {report_path}")
        
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        details = report.get("details", [])
        
        # Filter failed and partial segments
        failed_segments = [
            seg for seg in details
            if seg.get("validation", {}).get("status") in ["FAIL", "PARTIAL"]
        ]
        
        if not failed_segments:
            logger.info("No failed or partial segments found. Skipping analysis.")
            summary_report = {
                "total_analyzed": 0,
                "failed_count": 0,
                "partial_count": 0,
                "analyses": [],
                "overall_summary": "All segments passed validation!"
            }
        else:
            logger.info(f"Found {len(failed_segments)} failed/partial segments to analyze")
            
            analyses = []
            failed_count = 0
            partial_count = 0
            
            for seg in failed_segments:
                status = seg.get("validation", {}).get("status", "UNKNOWN")
                if status == "FAIL":
                    failed_count += 1
                elif status == "PARTIAL":
                    partial_count += 1
                
                logger.info(f"Analyzing {seg.get('segment_id')} ({status})")
                analysis = self.analyze_segment_failure(seg)
                analyses.append(analysis)
            
            # Generate overall summary
            overall_summary = self._generate_overall_summary(analyses, failed_count, partial_count)
            
            summary_report = {
                "total_analyzed": len(failed_segments),
                "failed_count": failed_count,
                "partial_count": partial_count,
                "analyses": analyses,
                "overall_summary": overall_summary
            }
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Failure analysis saved to {output_path}")
        self._print_summary(summary_report)
        
        return summary_report
    
    def _generate_overall_summary(self, analyses: List[Dict], failed_count: int, partial_count: int) -> str:
        """Generate overall summary from individual analyses"""
        
        prompt = f"""Analyze these {len(analyses)} Knowledge Graph validation failures and provide a brief overall summary.

**FAILED SEGMENTS:** {failed_count}
**PARTIAL SEGMENTS:** {partial_count}

**INDIVIDUAL ANALYSES:**
"""
        for analysis in analyses:
            prompt += f"\n{analysis['segment_id']} ({analysis['status']}, Overall: {analysis['scores']['overall']:.2f}):\n{analysis['analysis']}\n"
        
        prompt += """
**TASK:**
Provide a 2-3 paragraph overall summary covering:
1. Common patterns in failures (e.g., missing PDF evidence, hallucinated relationships, entity extraction issues)
2. Most critical issues to address
3. High-level recommendations to improve KG quality

Be concise and actionable."""
        
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to generate overall summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def _print_summary(self, report: Dict):
        """Print summary to console"""
        print("\n" + "=" * 70)
        print("FAILURE ANALYSIS REPORT")
        print("=" * 70)
        print(f"Total Analyzed:  {report['total_analyzed']}")
        print(f"Failed:          {report['failed_count']}")
        print(f"Partial:         {report['partial_count']}")
        print("=" * 70)
        print("\nOVERALL SUMMARY:")
        print(report['overall_summary'])
        print("=" * 70)


def main():
    """Main entry point for standalone execution"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_failures.py <validation_report.json> [output_file.json]")
        sys.exit(1)
    
    report_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "failure_analysis_report.json"
    
    # Configuration (should match crossKG.py)
    NVIDIA_API_KEY = "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"
    NIM_MODEL = "deepseek-ai/deepseek-v3.1"
    NIM_API_BASE = "https://integrate.api.nvidia.com/v1"
    
    analyzer = FailureAnalyzer(NVIDIA_API_KEY, NIM_MODEL, NIM_API_BASE)
    analyzer.analyze_report(report_path, output_path)


if __name__ == "__main__":
    main()
