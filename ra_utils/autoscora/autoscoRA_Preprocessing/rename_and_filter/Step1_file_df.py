import SimpleITK as sitk
import numpy as np
import os
import itertools
# import sys
import pandas as pd
# import re
import ntpath
import datetime


print(datetime.datetime.now())

# server data
dicom_server = "/project/autoscora/autoscoRA_images/autoscoRA_original_images"
output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output"
data_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/data"

# sample data
# output_folder = "/mnthome/autoscoRA/autoscoRA_Preprocessing/output/test"
# output_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/output/test"
# dicom_server = "/mnthome/autoscoRA/autoscoRA_Preprocessing/sample_xrays"
# dicom_server = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/sample_xrays"
# data_folder = "/mnthome/autoscoRA/autoscoRA_Preprocessing/data"
# data_folder = "/home/cir/tdeimel/autoscoRA/autoscoRA_Preprocessing/data"

emptystr = True

# reading dicom files
print("START: reading dicom files...")
dicom_files = [os.path.join(root, x) for root, dirs, files in os.walk(dicom_server) for x in files
               if os.path.join(dicom_server, 'changed_metadata_dicoms') not in root]
dicom_files = [f for f in dicom_files if not os.path.split(f)[1].startswith('.')]
# dicom_files = ["{}{}".format(dicom_paths, dicom_files_) for dicom_paths, dicom_files_in zip(dicom_paths, dicom_files)]
# dicom_files_flat = list(itertools.chain.from_iterable(dicom_files))
# print(dicom_files_flat)
nr_of_files = len(dicom_files)
print("DONE: reading dicom files")
print("OUTPUT: nr of files: ", nr_of_files)

# Read meta data of files into a dataframe
print("START: creating dataframe of datafiles' metadata...")
tags_df = {'pat_id': '0010|0010',
           'study_date': '0008|0020',
           'accession_nr': '0008|0050',
           'modality': '0008|0060',
           'body_part_examined': '0018|0015',
           'laterality': '0020|0060',
           'view_position': '0018|5101',
           'series_description': '0008|103e',
           'series_description_code': '0008|103f',
           'study_description': '0008|1030',
           'pixel_spacing': '0028|0030',
           'imager_pixel_spacing': '0018|1164',
           'samples_per_pixel': '0028|0002',
           'rows': '0028|0010',
           'columns': '0028|0011',
           'photometric_interpretation': '0028|0004'}

tags_dict_df = {key: np.nan for key in tags_df.keys()}

if 'pat_df' in globals():
    del pat_df
# i = dicom_files[0]
for i in range(0, len(dicom_files)):  # loop through image files
    # create reader object for current dicom file
    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_files[i])
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    tags = reader.GetMetaDataKeys()

    # read metadata into dictionary
    # dict_pat = {md_name: reader.GetMetaData(md_tag).encode("utf-8", "replace")
    #             for md_name, md_tag in tags_df.items() if md_tag in tags}
    # dict_pat = {md_name: reader.GetMetaData(md_tag).encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    #             for md_name, md_tag in tags_df.items() if md_tag in tags}
    # dict_pat = {md_name: (reader.GetMetaData(md_tag).encode("utf-8", "replace").decode("utf-8", "replace"))
    #             for md_name, md_tag in tags_df.items() if md_tag in tags}
    dict_pat = {md_name: (reader.GetMetaData(md_tag).encode("utf-8", "replace").decode("utf-8", "replace").strip()
                          if md_tag in tags else np.nan)
                for md_name, md_tag in tags_df.items()}  # strip() deletes leading and trailing white spaces

    # check that no missing tags (which might get final dataframe row out of order
    if len(dict_pat) != len([md_name for md_name, md_tag in tags_df.items()]):
        raise("missing tags in file" + dicom_files[i])

    # include old file name in dict
    dict_pat['filepathname_old'] = dicom_files[i]
    dict_pat['filepath_old'] = ntpath.split(dicom_files[i])[0]
    dict_pat['filename_old'] = ntpath.split(dicom_files[i])[1]

    # turn dict into df
    i_df = pd.DataFrame(dict_pat, index=[i], columns=dict_pat.keys())

    # append df for this image file to df for all image files
    if 'pat_df' not in globals():
        pat_df = i_df
    else:
        pat_df = pat_df.append(i_df)

    if i == len(dicom_files)-1:
        print("DONE: creating dataframe of datafiles' metadata")

# check that all old filenames are unique
unique_entries_filename = len(list(dict.fromkeys(pat_df['filename_old'])))
if unique_entries_filename == pat_df.shape[0]:
    print('GOOD: all old filenames seem to be unique!')
else:
    print('BAD: not all old filenames seem to be unique!')

# replace '' by 'emptystr'
if emptystr is True:
    print("START: replacing '' by 'emptystr'...")
    list_empty_idx = [[(i, j) for i in np.where(pat_df.iloc[:, j] == '')[0]] for j in range(0, pat_df.shape[1])
                      if (type(pat_df.iloc[0, j]) == str and np.where(pat_df.iloc[:, j] == '')[0].size > 0)]
    empty_idx = list(itertools.chain.from_iterable(list_empty_idx))
    all_empty = all([j == '' for j in [pat_df.iloc[empty_idx[i]] for i in range(0, len(empty_idx))]])
    if all_empty:
        print("GOOD: all pat_df entries identified as '' are in fact ''!")
    else:
        print("BAD: not all pat_df entries identified as '' are in fact ''!")
        bad_entries = [pat_df.iloc[empty_idx[i]] for i in range(0, len(empty_idx))
                       if pat_df.iloc[empty_idx[i]] != '']  # not yet tested!
    for i in empty_idx:
        pat_df.iloc[i] = 'emptystr'
    print("DONE: replacing '' by 'emptystr'")

# save pat_df
print("START: saving metadata dataframe...")
pat_df.to_pickle(output_folder + os.sep + "metadata_df.pkl")
pat_df.to_csv(output_folder + os.sep + "metadata_df.csv")
# df = pd.read_pickle("output_folder + os.sep + "metadata_df.pkl")
print("DONE: saving metadata dataframe")

# count files per unique patient+study_date combination --> can use this as well to identify double-hand/foot images!
print("START: saving nr of images per id+date combo...")
im_per_iddate = pat_df.groupby(['pat_id', 'study_date']).size().reset_index().rename(columns={0: 'count'})
im_per_iddate.to_pickle(output_folder + os.sep + "im_per_iddate.pkl")
im_per_iddate.to_csv(output_folder + os.sep + "im_per_iddate.csv")
print("DONE: saving nr of images per id+date combo")

# count number of unique patient+study_date combinations
iddate_count = len(im_per_iddate)
print("OUTPUT: number of id+date combinations:", iddate_count)

# dictionary of unique metadata entries
print("START: creating and saving df of unique metadata entries...")
pat_dict = pat_df.to_dict('series')
pat_dict_unique = {key: list(dict.fromkeys([i for i in pat_dict[key] if not pd.isnull(i)]))
                   for key, value in pat_dict.items() if key not in ["filepath_old", "filename_old",
                                                                     "filepathname_old", "columns",
                                                                     "rows", "pat_id", "study_date"]}
pat_dict_unique_maxlen = max([len(list(dict.fromkeys(value))) for key, value in pat_dict_unique.items()])
pat_dict_unique_nan = {key: value + [np.nan]*(pat_dict_unique_maxlen - len(value))
                       for key, value in pat_dict_unique.items()}
# if emptystr == True:
#    pat_dict_unique_nan = {key: [i if i != '' else 'emptystr' for i in value]
#                           for key, value in pat_dict_unique_nan.items()}

# dataframe of unique metadata entries
pat_dict_unique_df = pd.DataFrame(pat_dict_unique_nan, columns=pat_dict_unique_nan.keys())
pat_dict_unique_df.to_pickle(output_folder + os.sep + "metadata_unique_df.pkl")
pat_dict_unique_df.to_csv(output_folder + os.sep + "metadata_unique_df.csv")
# df = pd.read_pickle(output_folder + os.sep + "metadata_df.pkl")
print("DONE: creating and saving df of unique metadata entries...")

# all(pd.read_csv("output_folder + os.sep + "metadata_unique_df.csv", index_col=0) == pat_dict_unique_df)


# read accession numbers from Prinz
print("START: reading accession numbers from Prinz...")
prinz_acc_filename = data_folder + os.sep + 'pseudo_accession_Prinz-ad-CIR.txt'
accession_nrs_prinz = pd.read_csv(prinz_acc_filename,
                                  header=None)
print("DONE: reading accession numbers from Prinz")

# compare accession numbers from Prinz with those in data
print("START: comparing accession numbers in data with those received from Prinz...")
accession_nrs_prinz = accession_nrs_prinz.iloc[:, 0].tolist()
accession_nrs_data = pat_df['accession_nr'].tolist()
all_data_in_prinz = set(accession_nrs_data).issubset(accession_nrs_prinz)
all_prinz_in_data = set(accession_nrs_prinz).issubset(accession_nrs_data)
print("data acc nrs are subset of Prinz acc nrs:", all_data_in_prinz)
print("Prinz acc nrs are subset of data acc nrs:", all_prinz_in_data)
accession_nrs_inprinz_notindata = [x for x in accession_nrs_prinz if x not in accession_nrs_data]
accession_nrs_notinprinz_indata = [x for x in accession_nrs_data if x not in accession_nrs_prinz]
if not all_prinz_in_data:
    CT_MR_US = ['E3F3A5D0E0C6F3AC64262D7DF8AF68F19E9B3EFF3BE0F523798A2B71981B1886',
                '1EF1E51D2126DC2C582171F006C3D0580A84852F318F73B0E32A5693CCB63306',
                '6FD90EE795477B4D1D9F5C109D60C6A40365F76EA942430F68D65F3C714E9950',
                '4201B583C77A9D951DCC0E3869F6528D18FAC5C97E60482C0514AC5112200BE3',
                'B5627A62D6387E742EC9BC25378720EC7546351BF18BBC060ED4AAB5698450A6',
                'F2D863E40CABE576FA2EB2D31F2B7BE6E3EF00CE0C79C35D6C4A01C595B358FA',
                'F414FD46055D125A3420BA46134C848B3091E504EA9A6F28E092CC9B3772AA56',
                '1E5C0A088A5B4DB59840E94BFB799430C3E5A1BEF41647EBFD31635F643143D5',
                'A96ABEA585FE40B577F8582EA80165C373DA280F7189C20B50F6847B53E06279',
                '4E62EDAFFEB0CA70A639E562F8C0768B2C196FDD3F3285696DE0FDDE8B5B5958',
                'B733E083787DBF7EAFB910D2D441AD6179A1DF60AB435ECDD578F4E15CF781AC',
                '3D7780ADBE5D511CA55F5653D23AEF21E252988C12FCEF3CE70C8461A9F51C47',
                '5230AEA1AF6600C3DBBC7EFC87F22C2FBC9970591ECE5D17BA0B03F8098D5EC3',
                '58AFA84818AFB405B4DB8AFDD7A1ED12766DBDEEA1C485525E200C3C4F5271AD',
                'D7C9F5B1B2FA40D7466B983EDD106489A79CB349B50D5D33967701360BDF232D',
                '1EB60141B30525D1552A7AE9DE84B9671AF29FD5414B9E76573F782C8AEF4DB2',
                'FF56A2C3A583F3A9A90E96A178D349A4547C97B0D83A34499CEB3C64AB4F7C6D',
                '545FA4802A8C7F3B7B2DE2FC7C76EBB382BBA5F6CF65A6134CF00144A862B329',
                '8D0C5CE871D8D27F41E9BB4CB79AB7914CFFF4AB06B637F9750913EAD7BEC805',
                'D762F829B6BF59F0405C4E8DC45430CACC52B55B7BD1B50804C343BE8F2E0E90',
                'A1DA5219B822A1EB502CDAE48BEF783526F7B89AF6D65DA41F5406BA042ACF3C',
                '08C76F04E988E3AB97561FAB6EA89D42F0D65348947DC9A309D69F64241ECF45',
                'A94D79DCC8A4E09F5A714EBE968F4BCB7376E480AC5980DDEDF25BB4ADCFFA9A',
                '7D04C262C08C295EE260435786DDD426BB883A59DEC28B1262CD68EB334FAD28',
                'CB0AF3DB63B50B076C56DDE87967C78667E16D9A4F0C9D3CA722D51E552ED874',
                'A9F9E1BB2D041543BF2B0D4A5513F910948232DE694B2950029BC180FE59832C',
                '95440ED801B10FF1F20A50E07EFEA0A94B46CD3B556EBDE399C1F69E7865827A',
                '9D63B7DF7B84E51267F56D5AB56FFF971FCD96A1AA765BAC6415D5A7BEE886D5',
                'E8D69AE9D69CD49917C7A3A65F52507C87AF2B5FBF2D6A3CB26329F4DC17F649',
                'D88C533DCE79D54741FA85B5C965AFB199513A641441CD151247892286064969',
                '392F7D3BBEAF1130175AB988926CC08F358F35B1894FBEB448F2BADE7815DF8D',
                'AEA5A0EA7904D01DCCC3EB0F1F96D60ED422E19774FDB721EE648271360687B8',
                '2225A8174C672015AF249402D7308E22296D8DAE51D4C273C2FCEB3253011072',
                'B28877A92003B55791BE7848F388653E4557EE0415D1A1E4FD303517F6E0E734',
                'B9FA237D15A6CAA6273103227F0A7244F8E2D2417CE2BB4EEAE3D9E658DFBD2F',
                'EC5943CB09E2A487A7E1EF5AFD419008498D21399AA3E22E23F531FD560546CA',
                'B6ACF31BA078D6E602036363C27104CF43727284DB9207EE907759676A1970EB',
                '9958ED5C1AED975F85522D35C05B14DFABD4CB0D5C6844B2DE1FCAC4453C1A20',
                '0685B811BECCB4B78AA4BF921DC9E263C6573C5FF08B08E8D417012791BD21A4',
                '9F5113B231FCBD7D67111958A95D5409433AB4BC88797C386EEA1DDAA95FCE13',
                'DA94FB19D3F71BC6D5AE370D452D54FB6570693C5CAAAA0D1B8C1FF8DFAEB3E7',
                '3BC2C86C6923A50FD03C4D67D590459FFD4F3CECE9881080DE9FBE1CF2D32691',
                'C3A43810F0CB0D58E6549F5D8E47E2E0D2316225C2045DFDE3334EA32C3F737F',
                '35C825B99ADE9E477EEE5B4A53D9170800D345E6568F57B10A54A9C4F2140AEE',
                '2BB3F79FB57733BDD251756340D3D5B055C4B3DA4166498F1EC03C028C902DA7',
                'F2A38CFE4216107AE3F2CCEB98D9190774B4EC72993CB6AE0EB65F32AB79BE58',
                '9D0503F7C1E35FA4F8EFC38C832BA71D887753FB83CA8F7A3D7DE97895116353',
                '53201AB1B93D44A6B03FAEEE9456938B60254E35CB0C1955FCE086C9DE0F0E54',
                '12DB44B4DD41B434E8C13722834C009A14E2F0C3DCBB041AA178CD8836C7B925',
                'B5019D30AC79C1EB140776E21388C4AF732D67D40B6473CA39F65C1F3DA678B0',
                '12E16C88A15CBF93A73B060C11617AF54E529B7C93BF2AC7827A1D2C4992D3C1',
                '7BC5870777177AFABB51D00BDC1408828AD3FC81BF0D5B7C41AB457B657DE29B',
                '3BCF0A45E3C74C1B89A83D5FE7153C109C945ABC2E0FF0C835CA6E8BBF6E7EE4',
                'DA9E0F57D65CBEFF8CB31E503DD7F2D95A21153D466275BCBD5C1837CF278BC1',
                '24792E662F71DC629DE31943960B096F16077BF9949F4177CB20E9A2EAB130A4',
                '5A91EBFD43E3CBBB5DB36FE5198D5A6825B5A97A64BFD2F502922B8B01FE8296',
                '45A499E5FA56D266DF7DD81C6EAE83191A429B6A3F4E61F6A61847B2AA9D9B64',
                'DF10C055C2C8B8E9827BE92727B431465FB3D0A8A7980961041CCEBCB651FF83',
                '09B94CD6A9F7AD41147CD65F6BE58087E5FE4E5C88E9C4844E4FC34DE9290F7A',
                '5CB88CF9B715E9342FDFC46A0D2FE7CF1EB25C8B07D16A861F3C80552E2CA4B4',
                'C2DFC9C436CF8D3144FEB7DFD0CED98AD44E43BE8F25B34057210CECAB70FA8F']
    accession_nrs_inprinz_notindata_notCTMRUS = [x for x in accession_nrs_inprinz_notindata if x not in CT_MR_US]
    print("OUTPUT: accession nrs missing in data but present in Prinz (and not known CT, MR, US):")
    print(accession_nrs_inprinz_notindata_notCTMRUS)
if not all_data_in_prinz:
    print("OUTPUT: accession nrs missing in Prinz but present in data:")
    print(accession_nrs_notinprinz_indata)
print("DONE: comparing accession numbers in data with those received from Prinz...")

# upon user input, print head of metadata dataframe pat_df
# show_pat_df_input = str(input("press 'y' if you want to view head of "
#                               "metadata dataframe (else: press any other letter)"))
# if show_pat_df_input == 'y':
#    print(pat_df.head(n=10))
print(pat_df.head(n=10))

print(datetime.datetime.now())
