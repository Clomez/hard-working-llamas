import subprocess

########################################
# Run multiple scripts in sequence. 
# good for testing
########################################
program_list = ['run-10.py', 'run-20.py']

for program in program_list:
    subprocess.call(['python3', program])
    print("Finished:" + program)