# could also consider using:
# https://www.kaggle.com/skooch/cifar-10-in-tensorflow


ssh -i ~/.ssh/USEast21Aug_secret.pem ec2-user@ec2-34-227-49-179.compute-1.amazonaws.com

# map a folder to iMac
sudo sshfs -o allow_other,defer_permissions,IdentityFile=/Users/graham/.ssh/USEast21Aug_secret.pem ec2-user@ec2-34-227-49-179.compute-1.amazonaws.com:/home/ec2-user/remote /Users/graham/Documents/UIUC/1_CS498_Applied_Machine_Learning/hw9/remote

