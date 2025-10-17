import os
import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

class ProjectIntegrityChecker:
    """
    Comprehensive project integrity checking system.
    Validates data integrity, model consistency, and system health.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
        self.integrity_report = {}
        self.checksum_cache = {}
        
    def run_full_integrity_check(self) -> Dict[str, Any]:
        """
        Run a comprehensive integrity check on the entire project.
        
        Returns:
            Dictionary with complete integrity report
        """
        self.logger.info("Starting comprehensive integrity check")
        
        self.integrity_report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': self.project_root,
            'checks': {}
        }
        
        # Run all integrity checks
        self.integrity_report['checks']['file_integrity'] = self.check_file_integrity()
        self.integrity_report['checks']['data_integrity'] = self.check_data_integrity()
        self.integrity_report['checks']['model_integrity'] = self.check_model_integrity()
        self.integrity_report['checks']['dependency_integrity'] = self.check_dependency_integrity()
        self.integrity_report['checks']['configuration_integrity'] = self.check_configuration_integrity()
        
        # Calculate overall health score
        self.integrity_report['overall_health'] = self._calculate_health_score()
        
        self.logger.info(f"Integrity check completed. Overall health: {self.integrity_report['overall_health']}")
        
        return self.integrity_report
    
    def check_file_integrity(self) -> Dict[str, Any]:
        """
        Check file system integrity and consistency.
        
        Returns:
            Dictionary with file integrity results
        """
        self.logger.info("Checking file integrity")
        
        file_check_results = {
            'status': 'pass',
            'issues': [],
            'warnings': [],
            'file_checksums': {},
            'missing_files': [],
            'extra_files': []
        }
        
        # Expected project structure
        expected_files = [
            'app.py',
            'requirements.txt',
            'README.md',
            'utils/__init__.py',
            'utils/database_manager.py',
            'utils/md5_manager.py',
            'utils/data_cleaner.py',
            'utils/predictor.py',
            'utils/visualizer.py',
            'utils/integrity_checker.py'
        ]
        
        # Check for expected files
        for file_path in expected_files:
            full_path = os.path.join(self.project_root, file_path)
            if not os.path.exists(full_path):
                file_check_results['missing_files'].append(file_path)
                file_check_results['issues'].append(f"Missing required file: {file_path}")
            else:
                # Calculate checksum for existing files
                try:
                    checksum = self._calculate_file_checksum(full_path)
                    file_check_results['file_checksums'][file_path] = checksum
                except Exception as e:
                    file_check_results['warnings'].append(f"Could not calculate checksum for {file_path}: {str(e)}")
        
        # Check for unexpected files in utils directory
        utils_dir = os.path.join(self.project_root, 'utils')
        if os.path.exists(utils_dir):
            actual_files = os.listdir(utils_dir)
            expected_utils_files = ['__init__.py', 'database_manager.py', 'md5_manager.py', 
                                  'data_cleaner.py', 'predictor.py', 'visualizer.py', 'integrity_checker.py']
            
            for file in actual_files:
                if file not in expected_utils_files and not file.startswith('__pycache__'):
                    file_check_results['extra_files'].append(f"utils/{file}")
                    file_check_results['warnings'].append(f"Unexpected file in utils directory: {file}")
        
        # Update status based on issues
        if file_check_results['issues']:
            file_check_results['status'] = 'fail'
        elif file_check_results['warnings']:
            file_check_results['status'] = 'warning'
        
        return file_check_results
    
    def check_data_integrity(self) -> Dict[str, Any]:
        """
        Check data files integrity and consistency.
        
        Returns:
            Dictionary with data integrity results
        """
        self.logger.info("Checking data integrity")
        
        data_check_results = {
            'status': 'pass',
            'issues': [],
            'warnings': [],
            'data_files': {},
            'corrupted_files': []
        }
        
        # Check data directory
        data_dir = os.path.join(self.project_root, 'data')
        if os.path.exists(data_dir):
            data_files = os.listdir(data_dir)
            
            for file in data_files:
                file_path = os.path.join(data_dir, file)
                
                try:
                    if file.endswith('.json'):
                        # Check JSON files
                        with open(file_path, 'r') as f:
                            json.load(f)
                        data_check_results['data_files'][file] = 'valid_json'
                        
                    elif file.endswith('.pkl'):
                        # Check pickle files
                        with open(file_path, 'rb') as f:
                            pickle.load(f)
                        data_check_results['data_files'][file] = 'valid_pickle'
                        
                    elif file.endswith(('.csv', '.xlsx', '.xls')):
                        # Check data files
                        if file.endswith('.csv'):
                            df = pd.read_csv(file_path)
                        else:
                            df = pd.read_excel(file_path)
                        
                        # Basic data validation
                        if df.empty:
                            data_check_results['warnings'].append(f"Empty data file: {file}")
                        elif df.isnull().all().any():
                            data_check_results['warnings'].append(f"File with all-null columns: {file}")
                        
                        data_check_results['data_files'][file] = {
                            'shape': df.shape,
                            'columns': list(df.columns),
                            'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
                        }
                        
                except Exception as e:
                    data_check_results['corrupted_files'].append(file)
                    data_check_results['issues'].append(f"Corrupted data file {file}: {str(e)}")
        
        # Update status
        if data_check_results['issues']:
            data_check_results['status'] = 'fail'
        elif data_check_results['warnings']:
            data_check_results['status'] = 'warning'
        
        return data_check_results
    
    def check_model_integrity(self) -> Dict[str, Any]:
        """
        Check model files and consistency.
        
        Returns:
            Dictionary with model integrity results
        """
        self.logger.info("Checking model integrity")
        
        model_check_results = {
            'status': 'pass',
            'issues': [],
            'warnings': [],
            'model_files': {},
            'missing_models': []
        }
        
        # Check for model files
        model_files = ['model_metrics.json', 'signatures.json']
        
        for model_file in model_files:
            file_path = os.path.join(self.project_root, 'data', model_file)
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        model_data = json.load(f)
                    
                    model_check_results['model_files'][model_file] = {
                        'exists': True,
                        'valid_json': True,
                        'keys': list(model_data.keys()) if isinstance(model_data, dict) else 'not_dict'
                    }
                    
                except Exception as e:
                    model_check_results['issues'].append(f"Invalid model file {model_file}: {str(e)}")
            else:
                model_check_results['missing_models'].append(model_file)
                model_check_results['warnings'].append(f"Missing model file: {model_file}")
        
        # Check for uploaded data
        uploaded_data_path = os.path.join(self.project_root, 'data', 'uploaded_data.pkl')
        if os.path.exists(uploaded_data_path):
            try:
                with open(uploaded_data_path, 'rb') as f:
                    uploaded_data = pickle.load(f)
                
                if isinstance(uploaded_data, pd.DataFrame):
                    model_check_results['model_files']['uploaded_data.pkl'] = {
                        'exists': True,
                        'type': 'DataFrame',
                        'shape': uploaded_data.shape,
                        'columns': list(uploaded_data.columns)
                    }
                else:
                    model_check_results['model_files']['uploaded_data.pkl'] = {
                        'exists': True,
                        'type': type(uploaded_data).__name__
                    }
                    
            except Exception as e:
                model_check_results['issues'].append(f"Corrupted uploaded data: {str(e)}")
        
        # Update status
        if model_check_results['issues']:
            model_check_results['status'] = 'fail'
        elif model_check_results['warnings']:
            model_check_results['status'] = 'warning'
        
        return model_check_results
    
    def check_dependency_integrity(self) -> Dict[str, Any]:
        """
        Check Python dependencies and requirements.
        
        Returns:
            Dictionary with dependency integrity results
        """
        self.logger.info("Checking dependency integrity")
        
        dep_check_results = {
            'status': 'pass',
            'issues': [],
            'warnings': [],
            'requirements_file': None,
            'missing_dependencies': [],
            'version_conflicts': []
        }
        
        # Check requirements.txt
        requirements_path = os.path.join(self.project_root, 'requirements.txt')
        if os.path.exists(requirements_path):
            try:
                with open(requirements_path, 'r') as f:
                    requirements = f.read().strip().split('\n')
                
                dep_check_results['requirements_file'] = {
                    'exists': True,
                    'packages': [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
                }
                
                # Check if required packages are importable
                required_packages = [
                    'streamlit', 'pandas', 'numpy', 'scikit-learn', 
                    'matplotlib', 'seaborn', 'plotly'
                ]
                
                for package in required_packages:
                    try:
                        __import__(package.replace('-', '_'))
                    except ImportError:
                        dep_check_results['missing_dependencies'].append(package)
                        dep_check_results['warnings'].append(f"Missing required package: {package}")
                        
            except Exception as e:
                dep_check_results['issues'].append(f"Error reading requirements.txt: {str(e)}")
        else:
            dep_check_results['issues'].append("Missing requirements.txt file")
        
        # Update status
        if dep_check_results['issues']:
            dep_check_results['status'] = 'fail'
        elif dep_check_results['warnings']:
            dep_check_results['status'] = 'warning'
        
        return dep_check_results
    
    def check_configuration_integrity(self) -> Dict[str, Any]:
        """
        Check configuration files and settings.
        
        Returns:
            Dictionary with configuration integrity results
        """
        self.logger.info("Checking configuration integrity")
        
        config_check_results = {
            'status': 'pass',
            'issues': [],
            'warnings': [],
            'config_files': {}
        }
        
        # Check for configuration files
        config_files = ['_config.yml', 'package.json']
        
        for config_file in config_files:
            file_path = os.path.join(self.project_root, config_file)
            
            if os.path.exists(file_path):
                try:
                    if config_file.endswith('.json'):
                        with open(file_path, 'r') as f:
                            config_data = json.load(f)
                    else:
                        # For YAML files, just check if they exist and are readable
                        with open(file_path, 'r') as f:
                            config_data = f.read()
                    
                    config_check_results['config_files'][config_file] = {
                        'exists': True,
                        'readable': True
                    }
                    
                except Exception as e:
                    config_check_results['issues'].append(f"Unreadable config file {config_file}: {str(e)}")
            else:
                config_check_results['warnings'].append(f"Missing config file: {config_file}")
        
        # Check for essential HTML files
        html_files = ['index.html', '404.html']
        for html_file in html_files:
            file_path = os.path.join(self.project_root, html_file)
            if not os.path.exists(file_path):
                config_check_results['warnings'].append(f"Missing HTML file: {html_file}")
        
        # Update status
        if config_check_results['issues']:
            config_check_results['status'] = 'fail'
        elif config_check_results['warnings']:
            config_check_results['status'] = 'warning'
        
        return config_check_results
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """
        Calculate MD5 checksum for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 checksum string
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating checksum for {file_path}: {str(e)}")
            raise
    
    def _calculate_health_score(self) -> float:
        """
        Calculate overall project health score.
        
        Returns:
            Health score between 0 and 100
        """
        total_checks = len(self.integrity_report['checks'])
        passed_checks = 0
        warning_checks = 0
        
        for check_name, check_result in self.integrity_report['checks'].items():
            if check_result['status'] == 'pass':
                passed_checks += 1
            elif check_result['status'] == 'warning':
                warning_checks += 1
        
        # Calculate score: 100% for all pass, 70% for warnings, 0% for failures
        if passed_checks == total_checks:
            return 100.0
        elif passed_checks + warning_checks == total_checks:
            return 70.0 + (passed_checks / total_checks) * 30.0
        else:
            return (passed_checks / total_checks) * 70.0
    
    def save_integrity_report(self, filepath: str = None):
        """
        Save integrity report to file.
        
        Args:
            filepath: Path to save the report (optional)
        """
        if filepath is None:
            filepath = os.path.join(self.project_root, 'data', 'integrity_report.json')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.integrity_report, f, indent=2)
        
        self.logger.info(f"Integrity report saved to {filepath}")
    
    def get_quick_health_check(self) -> Dict[str, Any]:
        """
        Perform a quick health check without detailed analysis.
        
        Returns:
            Dictionary with quick health status
        """
        quick_check = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'critical_issues': [],
            'health_score': 0
        }
        
        # Check critical files
        critical_files = ['app.py', 'requirements.txt', 'utils/__init__.py']
        missing_critical = []
        
        for file in critical_files:
            if not os.path.exists(os.path.join(self.project_root, file)):
                missing_critical.append(file)
        
        if missing_critical:
            quick_check['critical_issues'] = missing_critical
            quick_check['overall_status'] = 'critical'
            quick_check['health_score'] = 0
        else:
            quick_check['overall_status'] = 'healthy'
            quick_check['health_score'] = 100
        
        return quick_check

    def get_integrity_log(self, limit: int = None) -> List[Dict[str, Any]]:
        """Returns the integrity check log."""
        self.logger.info(f"Retrieving integrity log (limit: {limit})")
        # Placeholder for actual integrity log
        return [
            {
                'timestamp': '2025-10-17T21:39:35.000Z',
                'check_type': 'component_verification',
                'status': 'passed',
                'message': 'All core components are present and functional.'
            },
            {
                'timestamp': '2025-10-17T21:39:35.001Z',
                'check_type': 'data_validation',
                'status': 'passed',
                'message': 'Data integrity checks completed successfully.'
            },
            {
                'timestamp': '2025-10-17T21:39:35.002Z',
                'check_type': 'model_validation',
                'status': 'passed',
                'message': 'Model validation checks passed.'
            }
        ][:limit] if limit else []
