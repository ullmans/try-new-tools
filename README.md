this project main goal is to try and get experiance with tools that i want to know how to work with.
for now, i learned how to work with git and github, with docker, and how to connect and use servers through  my ide (vs code).

the actions to open a project like this are:
1. make new project in vs code.
2. open the terminal (ctrl+ `) and connect to git (git init)
3. make a .gitignore file add write to it: __pycache__/
4. make a requirments.txt with the relevant libraries for the project (one in a raw)
5. make a dockerfile. check (try to build the image: docker build -t {inmage name} ." in the terminal)
6. make a new repo at github
7. connect the repo to my local project:
    git add .
    git commit -m "initial commit"
    git remote add origin <URL_OF_YOUR_REPO>
    git push -u origin main
  if the last one doesn't make it, try "git branch -M main" and then again.
8. enter my lightingAI account  and copy the SSH
9. connect the server to vs-code:
    crtl+shift+p (in vs-code)
    search for "remote SSH" (of this window)
    paste the SSH link
    choose linux
10. clone from github with the URL (can do it through vs-code)
11. build the imake for the docker (" docker build -t {image_name} .")
12. run the code ("docker run --gpus all {image_name}")

of course, there is the option not to use docker. in this case, just have to skip the irellevant step.

so far, i made the model to recognize digits (of MNIST dataset).
i would like to try and build a transformer and see what i can do with it
