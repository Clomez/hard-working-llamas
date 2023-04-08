import subprocess

########################################
# Run multiple scripts in sequence. 
# good for testing
########################################
program_list = ['script1.py', 'script2.py']

for program in program_list:
    subprocess.call(['python3', program])
    print("Finished:" + program)