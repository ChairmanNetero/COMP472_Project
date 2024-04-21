import os 
def Re_Labeling(path, file_name):
    print("start")
    # gives list of all files in the path directory
    files = os.listdir(path)

    # has a format to prevent errors in the file_name
    file_name = file_name.rstrip("_")

    # counter for iterating 
    counter = 509

    # For loop to go through each file and rename with the counter
    for file in files:
        # making the full path of the file
        file_proper_path = os.path.join(path, file)

        # checking if a file is present
        if os.path.isfile(file_proper_path):

            # get the file type to prevent a corruption of the jpg files
            _, file_type = os.path.splitext(file)

            # make the new file with the file_name, counter and the file type
            new_file = f"{counter}{file_name}{file_type}"
            counter += 1

            # making the full new file path
            new_file_proper_path = os.path.join(path, new_file)

            # renaming the file
            os.rename(file_proper_path, new_file_proper_path)
            print(new_file)

           
            


## first argument is the path of the directory, second argument is the the name you wanna give it 
##ENTER HERE FOR RUNNING THE SCRIPT 
Re_Labeling("../Final2", "_happy_senior_male")


print("done")