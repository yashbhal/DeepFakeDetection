from util import mkdir

# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets (point directly to the folder with class subfolders)
dataroot = './dataset/test'

# No subfolders, just the class folders inside test/
vals = ['']
multiclass = [0]

# model
model_path = 'weights/blur_jpg_prob0.5.pth'
