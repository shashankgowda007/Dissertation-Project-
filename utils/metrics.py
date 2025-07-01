import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
import time

class ModelMetrics:
    """
    Calculate and compare performance metrics for CNN and Vision Transformer models
    Provides clinical performance benchmarks and statistical comparisons
    """
    
    def __init__(self):
        # Clinical benchmarks from the ethics application
        self.target_sensitivity = 0.94
        self.target_specificity = 0.91
        self.target_processing_time = 15.0  # seconds
        self.radiologist_sensitivity = 0.89
        self.radiologist_specificity = 0.87
        
    def calculate_comparison_metrics(self, cnn_result, vit_result):
        """
        Calculate comprehensive comparison metrics between CNN and ViT models
        
        Args:
            cnn_result: CNN prediction result dictionary
            vit_result: ViT prediction result dictionary
            
        Returns:
            dict: Comprehensive metrics comparison
        """
        metrics = {
            'model_performance': self._calculate_individual_performance(cnn_result, vit_result),
            'agreement_analysis': self._calculate_agreement(cnn_result, vit_result),
            'clinical_benchmarks': self._compare_to_clinical_benchmarks(cnn_result, vit_result),
            'efficiency_metrics': self._calculate_efficiency_metrics(cnn_result, vit_result),
            'confidence_analysis': self._analyze_confidence_scores(cnn_result, vit_result),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return metrics
    
    def _calculate_individual_performance(self, cnn_result, vit_result):
        """Calculate individual model performance metrics"""
        return {
            'cnn': {
                'prediction': cnn_result['prediction'],
                'confidence': cnn_result['confidence'],
                'raw_score': cnn_result['raw_score'],
                'model_sensitivity': cnn_result['model_info']['sensitivity'],
                'model_specificity': cnn_result['model_info']['specificity'],
                'architecture': cnn_result['model_info']['architecture']
            },
            'vit': {
                'prediction': vit_result['prediction'],
                'confidence': vit_result['confidence'],
                'raw_score': vit_result['raw_score'],
                'model_sensitivity': vit_result['model_info']['sensitivity'],
                'model_specificity': vit_result['model_info']['specificity'],
                'architecture': vit_result['model_info']['architecture'],
                'attention_heads': vit_result['model_info']['num_heads']
            }
        }
    
    def _calculate_agreement(self, cnn_result, vit_result):
        """Calculate agreement between models"""
        prediction_agreement = cnn_result['prediction'] == vit_result['prediction']
        confidence_difference = abs(cnn_result['confidence'] - vit_result['confidence'])
        
        # Categorize agreement level
        if prediction_agreement and confidence_difference < 0.1:
            agreement_level = 'strong'
        elif prediction_agreement:
            agreement_level = 'moderate'
        else:
            agreement_level = 'weak'
        
        return {
            'prediction_agreement': prediction_agreement,
            'confidence_difference': confidence_difference,
            'agreement_level': agreement_level,
            'raw_score_correlation': self._calculate_score_correlation(
                cnn_result['raw_score'], vit_result['raw_score']
            ),
            'consensus_prediction': self._determine_consensus(cnn_result, vit_result)
        }
    
    def _calculate_score_correlation(self, cnn_score, vit_score):
        """Calculate correlation between model scores (simplified for single prediction)"""
        # For single predictions, we calculate a similarity metric
        score_difference = abs(cnn_score - vit_score)
        similarity = 1.0 - score_difference  # Higher similarity = lower difference
        return max(0.0, similarity)
    
    def _determine_consensus(self, cnn_result, vit_result):
        """Determine consensus prediction between models"""
        if cnn_result['prediction'] == vit_result['prediction']:
            # Same prediction - use weighted average confidence
            consensus_confidence = (cnn_result['confidence'] + vit_result['confidence']) / 2
            return {
                'prediction': cnn_result['prediction'],
                'confidence': consensus_confidence,
                'method': 'agreement'
            }
        else:
            # Different predictions - use model with higher confidence
            if cnn_result['confidence'] > vit_result['confidence']:
                return {
                    'prediction': cnn_result['prediction'],
                    'confidence': cnn_result['confidence'],
                    'method': 'cnn_higher_confidence'
                }
            else:
                return {
                    'prediction': vit_result['prediction'],
                    'confidence': vit_result['confidence'],
                    'method': 'vit_higher_confidence'
                }
    
    def _compare_to_clinical_benchmarks(self, cnn_result, vit_result):
        """Compare model performance to clinical benchmarks"""
        cnn_sens = cnn_result['model_info']['sensitivity']
        cnn_spec = cnn_result['model_info']['specificity']
        vit_sens = vit_result['model_info']['sensitivity']
        vit_spec = vit_result['model_info']['specificity']
        
        return {
            'target_benchmarks': {
                'sensitivity': self.target_sensitivity,
                'specificity': self.target_specificity,
                'processing_time': self.target_processing_time
            },
            'radiologist_baseline': {
                'sensitivity': self.radiologist_sensitivity,
                'specificity': self.radiologist_specificity
            },
            'cnn_vs_targets': {
                'sensitivity_vs_target': cnn_sens - self.target_sensitivity,
                'specificity_vs_target': cnn_spec - self.target_specificity,
                'sensitivity_vs_radiologist': cnn_sens - self.radiologist_sensitivity,
                'specificity_vs_radiologist': cnn_spec - self.radiologist_specificity,
                'meets_sensitivity_target': cnn_sens >= self.target_sensitivity,
                'meets_specificity_target': cnn_spec >= self.target_specificity
            },
            'vit_vs_targets': {
                'sensitivity_vs_target': vit_sens - self.target_sensitivity,
                'specificity_vs_target': vit_spec - self.target_specificity,
                'sensitivity_vs_radiologist': vit_sens - self.radiologist_sensitivity,
                'specificity_vs_radiologist': vit_spec - self.radiologist_specificity,
                'meets_sensitivity_target': vit_sens >= self.target_sensitivity,
                'meets_specificity_target': vit_spec >= self.target_specificity
            }
        }
    
    def _calculate_efficiency_metrics(self, cnn_result, vit_result):
        """Calculate computational efficiency metrics"""
        # Extract processing times from metadata if available
        cnn_time = getattr(cnn_result.get('processing_metadata', {}), 'processing_time', 2.0)
        vit_time = getattr(vit_result.get('processing_metadata', {}), 'processing_time', 1.5)
        
        return {
            'processing_times': {
                'cnn_seconds': cnn_time,
                'vit_seconds': vit_time,
                'total_seconds': cnn_time + vit_time,
                'fastest_model': 'cnn' if cnn_time < vit_time else 'vit'
            },
            'efficiency_vs_targets': {
                'cnn_meets_target': cnn_time <= self.target_processing_time,
                'vit_meets_target': vit_time <= self.target_processing_time,
                'combined_meets_target': (cnn_time + vit_time) <= self.target_processing_time,
                'cnn_time_vs_target': cnn_time - self.target_processing_time,
                'vit_time_vs_target': vit_time - self.target_processing_time
            },
            'throughput_estimates': {
                'cnn_cases_per_hour': 3600 / cnn_time if cnn_time > 0 else 0,
                'vit_cases_per_hour': 3600 / vit_time if vit_time > 0 else 0,
                'combined_cases_per_hour': 3600 / (cnn_time + vit_time) if (cnn_time + vit_time) > 0 else 0
            }
        }
    
    def _analyze_confidence_scores(self, cnn_result, vit_result):
        """Analyze confidence scores and decision boundaries"""
        cnn_conf = cnn_result['confidence']
        vit_conf = vit_result['confidence']
        cnn_raw = cnn_result['raw_score']
        vit_raw = vit_result['raw_score']
        
        return {
            'confidence_statistics': {
                'cnn_confidence': cnn_conf,
                'vit_confidence': vit_conf,
                'average_confidence': (cnn_conf + vit_conf) / 2,
                'confidence_difference': abs(cnn_conf - vit_conf),
                'higher_confidence_model': 'cnn' if cnn_conf > vit_conf else 'vit'
            },
            'raw_score_analysis': {
                'cnn_raw_score': cnn_raw,
                'vit_raw_score': vit_raw,
                'score_difference': abs(cnn_raw - vit_raw),
                'score_agreement': abs(cnn_raw - vit_raw) < 0.1
            },
            'decision_boundary_analysis': {
                'cnn_distance_from_threshold': abs(cnn_raw - 0.5),
                'vit_distance_from_threshold': abs(vit_raw - 0.5),
                'cnn_decision_confidence': 'high' if abs(cnn_raw - 0.5) > 0.3 else 'low',
                'vit_decision_confidence': 'high' if abs(vit_raw - 0.5) > 0.3 else 'low'
            },
            'clinical_confidence_assessment': self._assess_clinical_confidence(cnn_result, vit_result)
        }
    
    def _assess_clinical_confidence(self, cnn_result, vit_result):
        """Assess overall clinical confidence based on model agreement and performance"""
        prediction_agreement = cnn_result['prediction'] == vit_result['prediction']
        avg_confidence = (cnn_result['confidence'] + vit_result['confidence']) / 2
        confidence_difference = abs(cnn_result['confidence'] - vit_result['confidence'])
        
        # Determine confidence level
        if prediction_agreement and avg_confidence > 0.8 and confidence_difference < 0.1:
            confidence_level = 'very_high'
            recommendation = "Strong model agreement with high confidence supports clinical decision"
        elif prediction_agreement and avg_confidence > 0.6:
            confidence_level = 'high'
            recommendation = "Good model agreement supports clinical decision"
        elif prediction_agreement:
            confidence_level = 'moderate'
            recommendation = "Models agree but with moderate confidence - consider clinical correlation"
        elif avg_confidence > 0.7:
            confidence_level = 'moderate'
            recommendation = "Models disagree but individual confidence is high - review attention maps"
        else:
            confidence_level = 'low'
            recommendation = "Low confidence and/or disagreement - recommend additional evaluation"
        
        return {
            'overall_confidence_level': confidence_level,
            'clinical_recommendation': recommendation,
            'factors': {
                'model_agreement': prediction_agreement,
                'average_confidence': avg_confidence,
                'confidence_stability': confidence_difference < 0.15
            }
        }
    
    def generate_performance_summary(self, metrics):
        """Generate human-readable performance summary"""
        summary = {
            'executive_summary': self._create_executive_summary(metrics),
            'detailed_findings': self._create_detailed_findings(metrics),
            'clinical_recommendations': self._create_clinical_recommendations(metrics),
            'technical_notes': self._create_technical_notes(metrics)
        }
        
        return summary
    
    def _create_executive_summary(self, metrics):
        """Create executive summary of model performance"""
        agreement = metrics['agreement_analysis']
        benchmarks = metrics['clinical_benchmarks']
        
        summary_points = []
        
        # Model agreement
        if agreement['prediction_agreement']:
            summary_points.append(f"✅ Models agree on diagnosis ({agreement['agreement_level']} agreement)")
        else:
            summary_points.append(f"⚠️ Models disagree on diagnosis")
        
        # Performance vs targets
        cnn_meets_targets = (benchmarks['cnn_vs_targets']['meets_sensitivity_target'] and 
                           benchmarks['cnn_vs_targets']['meets_specificity_target'])
        vit_meets_targets = (benchmarks['vit_vs_targets']['meets_sensitivity_target'] and 
                           benchmarks['vit_vs_targets']['meets_specificity_target'])
        
        if cnn_meets_targets and vit_meets_targets:
            summary_points.append("✅ Both models meet clinical performance targets")
        elif cnn_meets_targets or vit_meets_targets:
            summary_points.append("⚠️ One model meets clinical targets")
        else:
            summary_points.append("❌ Models below clinical performance targets")
        
        return summary_points
    
    def _create_detailed_findings(self, metrics):
        """Create detailed performance findings"""
        performance = metrics['model_performance']
        benchmarks = metrics['clinical_benchmarks']
        
        findings = {
            'cnn_performance': {
                'sensitivity': f"{performance['cnn']['model_sensitivity']:.1%}",
                'specificity': f"{performance['cnn']['model_specificity']:.1%}",
                'vs_radiologist_sensitivity': f"+{benchmarks['cnn_vs_targets']['sensitivity_vs_radiologist']:.1%}",
                'vs_radiologist_specificity': f"+{benchmarks['cnn_vs_targets']['specificity_vs_radiologist']:.1%}"
            },
            'vit_performance': {
                'sensitivity': f"{performance['vit']['model_sensitivity']:.1%}",
                'specificity': f"{performance['vit']['model_specificity']:.1%}",
                'vs_radiologist_sensitivity': f"+{benchmarks['vit_vs_targets']['sensitivity_vs_radiologist']:.1%}",
                'vs_radiologist_specificity': f"+{benchmarks['vit_vs_targets']['specificity_vs_radiologist']:.1%}"
            }
        }
        
        return findings
    
    def _create_clinical_recommendations(self, metrics):
        """Create clinical recommendations based on analysis"""
        confidence = metrics['confidence_analysis']['clinical_confidence_assessment']
        efficiency = metrics['efficiency_metrics']
        
        recommendations = [
            confidence['clinical_recommendation']
        ]
        
        # Add efficiency recommendations
        if not efficiency['efficiency_vs_targets']['combined_meets_target']:
            recommendations.append("Consider optimizing processing pipeline for real-time clinical use")
        
        return recommendations
    
    def _create_technical_notes(self, metrics):
        """Create technical implementation notes"""
        efficiency = metrics['efficiency_metrics']
        
        notes = [
            f"Combined processing time: {efficiency['processing_times']['total_seconds']:.1f}s",
            f"Estimated throughput: {efficiency['throughput_estimates']['combined_cases_per_hour']:.0f} cases/hour"
        ]
        
        return notes
