import os

name_exe = "ltrs.exe"
name_mode = "prime"
name_params = "params"
name_inputfile = "prime_data_for_ltrs.txt"
name_outputfile = "output"


def get_file_path(name):
    return f"d:\Dropbox\~in-progress\\repo~Orthographic-DNN\external\LTRS\{name}"


if __name__ == "__main__":
    # os.system("external/LTRS/ltrs.exe ")
    os.system(
        f"{get_file_path(name_exe)} {name_mode} {get_file_path(name_params)} {get_file_path(name_inputfile)} {get_file_path(name_outputfile)}"
    )

    # d:\Dropbox\~in-progress\repo~Orthographic-DNN\external\LTRS\ltrs.exe prime d:\Dropbox\~in-progress\repo~Orthographic-DNN\external\LTRS\params d:\Dropbox\~in-progress\repo~Orthographic-DNN\external\LTRS\design d:\Dropbox\~in-progress\repo~Orthographic-DNN\external\LTRS\out
