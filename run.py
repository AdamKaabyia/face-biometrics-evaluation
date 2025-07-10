#!/usr/bin/env python3
"""
Face Biometrics Evaluation CLI
A comprehensive command-line interface for face recognition evaluation and morphing attacks.
"""

import os
import sys
import subprocess
import platform
import argparse
import shutil
import json
from pathlib import Path
from typing import List, Optional
import time

# ANSI color codes for better CLI experience
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class FaceBiometricsCLI:
    def __init__(self):
        self.is_windows = platform.system() == 'Windows'
        self.python_cmd = self._get_python_command()
        self.venv_name = 'face-biometrics-venv'
        self.data_dir = Path('Data')
        self.results_dir = Path('results')
        self.face_dir = self.data_dir / 'Face'

    def _get_python_command(self) -> str:
        """Get the appropriate Python command for the platform"""
        if self.is_windows:
            for cmd in ['python', 'python3']:
                if shutil.which(cmd):
                    return cmd
        else:
            for cmd in ['python3.11', 'python3', 'python']:
                if shutil.which(cmd):
                    return cmd
        return 'python'

    def _print_colored(self, message: str, color: str = Colors.ENDC) -> None:
        """Print colored message"""
        print(f"{color}{message}{Colors.ENDC}")

    def _print_header(self, title: str) -> None:
        """Print a formatted header"""
        self._print_colored(f"{title}", Colors.HEADER + Colors.BOLD)

    def _print_success(self, message: str) -> None:
        """Print success message"""
        self._print_colored(f"SUCCESS: {message}", Colors.OKGREEN)

    def _print_error(self, message: str) -> None:
        """Print error message"""
        self._print_colored(f"ERROR: {message}", Colors.FAIL)

    def _print_warning(self, message: str) -> None:
        """Print warning message"""
        self._print_colored(f"WARNING: {message}", Colors.WARNING)

    def _print_info(self, message: str) -> None:
        """Print info message"""
        self._print_colored(f"INFO: {message}", Colors.OKBLUE)

    def _run_command(self, cmd: str, shell: bool = True, check: bool = True, capture_output: bool = False) -> tuple:
        """Run a command with proper error handling"""
        try:
            if not capture_output:
                self._print_info(f"Running: {cmd}")

            if self.is_windows:
                result = subprocess.run(cmd, shell=shell, check=check,
                                      capture_output=capture_output, text=True)
            else:
                result = subprocess.run(cmd, shell=shell, check=check,
                                      executable='/bin/bash', capture_output=capture_output, text=True)

            return True, result.stdout if capture_output else ""
        except subprocess.CalledProcessError as e:
            self._print_error(f"Command failed: {e}")
            return False, e.stderr if capture_output else ""

    def _venv_exists(self) -> bool:
        """Check if virtual environment exists"""
        if self.is_windows:
            return (Path(self.venv_name) / 'Scripts' / 'python.exe').exists()
        else:
            return (Path(self.venv_name) / 'bin' / 'python').exists()

    def _get_venv_python(self) -> str:
        """Get the virtual environment Python executable path"""
        if self.is_windows:
            return str(Path(self.venv_name) / 'Scripts' / 'python.exe')
        else:
            return str(Path(self.venv_name) / 'bin' / 'python')

    def _activate_and_run(self, python_cmd: str) -> tuple:
        """Run a Python command in the virtual environment"""
        if not self._venv_exists():
            self._print_error("Virtual environment not found. Run 'python run.py setup' first.")
            return False, ""

        if self.is_windows:
            cmd = f"{self.venv_name}\\Scripts\\activate.bat && python {python_cmd}"
        else:
            cmd = f"source {self.venv_name}/bin/activate && python {python_cmd}"

        return self._run_command(cmd)

    def show_help(self) -> None:
        """Show comprehensive help information"""
        self._print_header("Face Biometrics Evaluation CLI")

        print(f"""
{Colors.OKBLUE}DESCRIPTION:{Colors.ENDC}
    A comprehensive face recognition evaluation system that compares multiple
    face recognition methods and evaluates their vulnerability to morphing attacks.

{Colors.OKBLUE}COMMANDS:{Colors.ENDC}
    {Colors.OKGREEN}setup{Colors.ENDC}          Create virtual environment and install dependencies
    {Colors.OKGREEN}install{Colors.ENDC}        Install/update dependencies
         {Colors.OKGREEN}check-data{Colors.ENDC}     Check if face dataset is available
              {Colors.OKGREEN}roc{Colors.ENDC}            Run face recognition evaluation
    {Colors.OKGREEN}html{Colors.ENDC}           Generate interactive HTML ROC curves
    {Colors.OKGREEN}morph{Colors.ENDC}          Run morph attack evaluation
     {Colors.OKGREEN}all{Colors.ENDC}            Run both roc and morph evaluations
    {Colors.OKGREEN}clean{Colors.ENDC}          Clean up temporary files
    {Colors.OKGREEN}reset{Colors.ENDC}          Clean everything (including venv and extracted data)
    {Colors.OKGREEN}results{Colors.ENDC}        Display recent results
    {Colors.OKGREEN}status{Colors.ENDC}         Check system and project status
    {Colors.OKGREEN}interactive{Colors.ENDC}    Start interactive mode
    {Colors.OKGREEN}help{Colors.ENDC}           Show this help message

{Colors.OKBLUE}USAGE:{Colors.ENDC}
    python run.py <command> [options]

{Colors.OKBLUE}EXAMPLES:{Colors.ENDC}
    python run.py setup              # First-time setup
    python run.py all                # Run complete evaluation
    python run.py status             # Check project status
    python run.py interactive        # Start interactive mode

{Colors.OKBLUE}SYSTEM INFO:{Colors.ENDC}
    OS: {platform.system()} {platform.release()}
    Python: {self.python_cmd}
    Architecture: {platform.architecture()[0]}
        """)

    def setup(self) -> bool:
        """Create virtual environment and install dependencies"""
        self._print_header("Setting Up Environment")

        if self._venv_exists():
            self._print_warning("Virtual environment already exists. Use 'install' to update dependencies.")
            return True

        # Create virtual environment
        self._print_info("Creating virtual environment...")
        success, _ = self._run_command(f"{self.python_cmd} -m venv {self.venv_name}")
        if not success:
            self._print_error("Failed to create virtual environment")
            return False

        # Install dependencies
        self._print_info("Installing dependencies...")
        if self.is_windows:
            cmd = f"{self.venv_name}\\Scripts\\activate.bat && python -m pip install --upgrade pip && pip install -r requirements.txt"
        else:
            cmd = f"source {self.venv_name}/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt"

        success, _ = self._run_command(cmd)
        if success:
            self._print_success("Environment setup complete!")
            self._print_info("You can now run: python run.py all")
            return True
        else:
            self._print_error("Setup failed")
            return False

    def install(self) -> bool:
        """Install/update dependencies"""
        self._print_header("Installing Dependencies")

        if not self._venv_exists():
            self._print_error("Virtual environment not found. Run 'setup' first.")
            return False

        if self.is_windows:
            cmd = f"{self.venv_name}\\Scripts\\activate.bat && python -m pip install --upgrade pip && pip install -r requirements.txt"
        else:
            cmd = f"source {self.venv_name}/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt"

        success, _ = self._run_command(cmd)
        if success:
            self._print_success("Dependencies updated!")
            return True
        else:
            self._print_error("Installation failed")
            return False

    def check_data(self) -> bool:
        """Check if face dataset is available"""
        self._print_header("Checking Face Dataset")

        if not self.face_dir.exists():
            self._print_error(f"Face dataset directory not found: {self.face_dir}")
            self._print_info("Please ensure face images are in Data/Face/ directory")
            self._print_info("Expected structure: Data/Face/person_id/image_files")
            return False

        # Count number of person directories
        person_dirs = [d for d in self.face_dir.iterdir() if d.is_dir()]
        if not person_dirs:
            self._print_error("No person directories found in Data/Face/")
            self._print_info("Please ensure face images are organized in person subdirectories")
            return False

        # Count total images
        total_images = 0
        for person_dir in person_dirs:
            images = [f for f in person_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in ['.bmp', '.jpg', '.jpeg', '.png']]
            total_images += len(images)

        self._print_success(f"Dataset found: {len(person_dirs)} persons, {total_images} images")
        return True

    def run_main(self) -> bool:
        """Run main face recognition evaluation"""
        self._print_header("Face Recognition Evaluation")

        if not self.check_data():
            return False

        self._print_info("Starting face recognition evaluation...")
        self._print_info("This may take 10-30 minutes depending on your system")

        success, _ = self._activate_and_run("roc.py")
        if success:
            self._print_success("Face recognition evaluation completed!")
            return True
        else:
            self._print_error("Face recognition evaluation failed")
            return False

    def run_html(self) -> bool:
        """Generate interactive HTML ROC curves"""
        self._print_header("Interactive HTML ROC Generation")

        if not self.check_data():
            return False

        self._print_info("Generating interactive HTML ROC curves...")
        self._print_info("This may take 10-30 minutes depending on your system")

        success, _ = self._activate_and_run("generate_html_roc.py")
        if success:
            self._print_success("Interactive HTML ROC curves generated!")
            self._print_info("Open results/index.html in your browser to view")
            return True
        else:
            self._print_error("HTML ROC generation failed")
            return False

    def run_morph(self) -> bool:
        """Run morph attack evaluation"""
        self._print_header("Morphing Attack Evaluation")

        if not self.check_data():
            return False

        self._print_info("Starting morphing attack evaluation...")
        self._print_info("This may take 5-15 minutes depending on your system")

        success, _ = self._activate_and_run("morph.py")
        if success:
            self._print_success("Morphing attack evaluation completed!")
            return True
        else:
            self._print_error("Morphing attack evaluation failed")
            return False

    def run_all(self) -> bool:
        """Run both evaluations"""
        self._print_header("Complete Evaluation Suite")

        self._print_info("Running complete face biometrics evaluation...")
        start_time = time.time()

        success = True
        if not self.run_main():
            success = False

        if success and not self.run_morph():
            success = False

        elapsed = time.time() - start_time
        if success:
            self._print_success(f"Complete evaluation finished in {elapsed/60:.1f} minutes!")
            self._print_info("Check results with: python run.py results")
        else:
            self._print_error("Evaluation failed")

        return success

    def clean(self) -> bool:
        """Clean up temporary files"""
        self._print_header("Cleaning Temporary Files")

        patterns_to_remove = [
            '__pycache__', '.pytest_cache', '.mypy_cache',
            '*.pyc', '*.log', 'temp_*.txt'
        ]

        for pattern in patterns_to_remove:
            if self.is_windows:
                if pattern.startswith('*'):
                    self._run_command(f'del /q {pattern}', check=False, capture_output=True)
                else:
                    self._run_command(f'rmdir /s /q {pattern}', check=False, capture_output=True)
            else:
                if pattern.startswith('*'):
                    self._run_command(f'find . -name "{pattern}" -delete', check=False, capture_output=True)
                else:
                    self._run_command(f'rm -rf {pattern}', check=False, capture_output=True)

        self._print_success("Temporary files cleaned")
        return True

    def reset(self) -> bool:
        """Clean everything including virtual environment and extracted data"""
        self._print_header("Resetting Project")

        # Ask for confirmation
        response = input(f"{Colors.WARNING}This will remove virtual environment and extracted data. Continue? (y/N): {Colors.ENDC}")
        if response.lower() != 'y':
            self._print_info("Reset cancelled")
            return False

        self.clean()

        # Remove virtual environment
        if Path(self.venv_name).exists():
            if self.is_windows:
                self._run_command(f'rmdir /s /q {self.venv_name}', check=False, capture_output=True)
            else:
                self._run_command(f'rm -rf {self.venv_name}', check=False, capture_output=True)

        # Remove extracted data
        face_dir = self.data_dir / 'Face'
        if face_dir.exists():
            if self.is_windows:
                self._run_command(f'rmdir /s /q "{face_dir}"', check=False, capture_output=True)
            else:
                self._run_command(f'rm -rf "{face_dir}"', check=False, capture_output=True)

        self._print_success("Project reset complete")
        self._print_info("Run 'python run.py setup' to start fresh")
        return True

    def show_results(self) -> None:
        """Display recent results"""
        self._print_header("Results Summary")

        if not self.results_dir.exists():
            self._print_warning("No results directory found")
            self._print_info("Run 'python run.py all' to generate results")
            return

        # List result files
        result_files = list(self.results_dir.glob('*'))
        if not result_files:
            self._print_warning("No result files found")
            return

        self._print_info("Result files:")
        for file in result_files:
            if file.is_file():
                size = file.stat().st_size / 1024  # KB
                modified = time.ctime(file.stat().st_mtime)
                print(f"  • {file.name} ({size:.1f} KB, {modified})")

        # Show main evaluation summary
        fmr_file = self.results_dir / 'fmr_fnmr_table.csv'
        if fmr_file.exists():
            print(f"\n{Colors.OKBLUE}Face Recognition Results:{Colors.ENDC}")
            with open(fmr_file, 'r') as f:
                lines = f.readlines()
                print("".join(lines[:min(6, len(lines))]))

        # Show morph attack summary
        morph_file = self.results_dir / 'morph_attack_summary.json'
        if morph_file.exists():
            print(f"\n{Colors.OKBLUE}Morphing Attack Results:{Colors.ENDC}")
            with open(morph_file, 'r') as f:
                data = json.load(f)
                print(f"  Total pairs tested: {data.get('total_pairs', 'N/A')}")
                results = data.get('results', {})
                for method, stats in results.items():
                    rate = stats.get('rate', 0)
                    count = stats.get('count', 0)
                    print(f"  {method}: {count} successful attacks ({rate:.1f}%)")

    def check_status(self) -> None:
        """Check system and project status"""
        self._print_header("System & Project Status")

        # System info
        print(f"{Colors.OKBLUE}System Information:{Colors.ENDC}")
        print(f"  OS: {platform.system()} {platform.release()}")
        print(f"  Python: {sys.version.split()[0]} ({sys.executable})")
        print(f"  Architecture: {platform.architecture()[0]}")

        # Disk space
        if self.is_windows:
            total, used, free = shutil.disk_usage('.')
            print(f"  Available space: {free // (1024**3):.1f} GB")
        else:
            success, output = self._run_command('df -h .', capture_output=True, check=False)
            if success:
                lines = output.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) >= 4:
                        print(f"  Available space: {parts[3]}")

        # Project status
        print(f"\n{Colors.OKBLUE}Project Status:{Colors.ENDC}")

        # Virtual environment
        if self._venv_exists():
            self._print_success("Virtual environment: Ready")
        else:
            self._print_error("Virtual environment: Not found")

        # Dataset
        if self.face_dir.exists():
            person_dirs = [d for d in self.face_dir.iterdir() if d.is_dir()]
            if person_dirs:
                total_images = sum(len([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in ['.bmp', '.jpg', '.jpeg', '.png']]) for d in person_dirs)
                self._print_success(f"Dataset: {len(person_dirs)} persons, {total_images} images")
            else:
                self._print_warning("Dataset directory exists but no person folders found")
        else:
            self._print_error("Dataset: Not found in Data/Face/")

        # Results
        if self.results_dir.exists():
            result_files = [f for f in self.results_dir.iterdir() if f.is_file()]
            if result_files:
                self._print_success(f"Results: {len(result_files)} files found")
            else:
                self._print_warning("Results: No files found")
        else:
            self._print_warning("Results: Directory not found")

        # Dependencies check
        if self._venv_exists():
            print(f"\n{Colors.OKBLUE}Checking Dependencies:{Colors.ENDC}")
            success, _ = self._activate_and_run("-c \"import torch, cv2, deepface, numpy; print('All dependencies OK')\"")
            if success:
                self._print_success("Dependencies: All required packages available")
            else:
                self._print_error("Dependencies: Some packages missing")

    def interactive_mode(self) -> None:
        """Start interactive mode"""
        self._print_header("Interactive Mode")

        commands = {
            '1': ('setup', 'Setup environment'),
            '2': ('roc', 'Run face recognition evaluation'),
            '3': ('html', 'Generate interactive HTML ROC curves'),
            '4': ('morph', 'Run morphing attack evaluation'),
            '5': ('all', 'Run complete evaluation'),
            '6': ('results', 'Show results'),
            '7': ('status', 'Check status'),
            '8': ('clean', 'Clean temporary files'),
            '9': ('reset', 'Reset project'),
            'q': ('quit', 'Quit interactive mode')
        }

        while True:
            print(f"\n{Colors.OKBLUE}Available Commands:{Colors.ENDC}")
            for key, (cmd, desc) in commands.items():
                print(f"  {key}) {desc}")

            choice = input(f"\n{Colors.OKCYAN}Enter command number (or 'q' to quit): {Colors.ENDC}").strip().lower()

            if choice == 'q':
                self._print_info("Goodbye!")
                break
            elif choice in commands:
                cmd = commands[choice][0]
                if cmd == 'quit':
                    break
                elif cmd == 'setup':
                    self.setup()
                elif cmd == 'roc':
                    self.run_main()
                elif cmd == 'html':
                    self.run_html()
                elif cmd == 'morph':
                    self.run_morph()
                elif cmd == 'all':
                    self.run_all()
                elif cmd == 'results':
                    self.show_results()
                elif cmd == 'status':
                    self.check_status()
                elif cmd == 'clean':
                    self.clean()
                elif cmd == 'reset':
                    self.reset()
            else:
                self._print_warning("Invalid choice. Please try again.")

    def run_command(self, command: str) -> bool:
        """Execute a specific command"""
        if command == 'help':
            self.show_help()
            return True
        elif command == 'setup':
            return self.setup()
        elif command == 'install':
            return self.install()
        elif command in ['check-data', 'check']:
            return self.check_data()
        elif command in ['roc', 'run-roc']:
            return self.run_main()
        elif command in ['html', 'run-html']:
            return self.run_html()
        elif command in ['morph', 'run-morph']:
            return self.run_morph()
        elif command in ['all', 'run-all']:
            return self.run_all()
        elif command == 'clean':
            return self.clean()
        elif command in ['reset', 'clean-all']:
            return self.reset()
        elif command in ['results', 'show-results']:
            self.show_results()
            return True
        elif command in ['status', 'check-system']:
            self.check_status()
            return True
        elif command == 'interactive':
            self.interactive_mode()
            return True
        else:
            self._print_error(f"Unknown command: {command}")
            self._print_info("Use 'python run.py help' to see available commands")
            return False


def main():
    cli = FaceBiometricsCLI()

    parser = argparse.ArgumentParser(
        description='Face Biometrics Evaluation CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py setup              # First-time setup
  python run.py all                # Run complete evaluation
  python run.py status             # Check project status
  python run.py interactive        # Start interactive mode
        """
    )

    parser.add_argument(
        'command',
        nargs='?',
        default='help',
        help='Command to execute (use "help" to see all commands)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Face Biometrics Evaluation CLI v1.0'
    )

    args = parser.parse_args()

    # If no command provided, show help
    if len(sys.argv) == 1:
        cli.show_help()
        return

    # Execute command
    try:
        success = cli.run_command(args.command)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        cli._print_warning("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        cli._print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()