import difflib
import pydicom
different_dcm_file =a
dcm_file = pydicom.dcmread(...)
different_dcm_file = pydicom.dcmread(...)

datasets = tuple((dcm_file, different_dcm_file))  # dcm_file_sitk))

# difflib compare functions require a list of lines, each terminated with
# newline character massage the string representation of each dicom dataset
# into this form:
rep = []
for dataset in datasets:
    lines = str(dataset).split("\n")
    lines = [line + "\n" for line in lines]  # add the newline to end
    rep.append(lines)

diff = difflib.Differ()
for line in diff.compare(rep[0], rep[1]):
    if line[0] != "?":
        print(line)
