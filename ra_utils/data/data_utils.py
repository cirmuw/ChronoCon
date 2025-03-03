


def extract_landmarks_from_df(dfm, image_idx=0):
    landmark_columns = dfm.filter(regex="^landmark").columns
    landmarks = dfm.loc[image_idx, landmark_columns].values.reshape(-1, 2)
    return landmarks




def extract_extras_from_filename(filename: str): 
    x = filename.split(".dcm")[0].split("_")
    d = {
        "filename": filename.replace(".dcm",""), 
        "id": x[0],
        "date_str": x[1],
        "sex": x[2],
        "left_or_right": x[3],
        # I dont know what the rest means:  dp_MTwo_InvNo_RotNo_BOk_OPNo_app_ComNo
        "x4": x[4],
        "x5": x[5],
        "x6": x[6],
        "x7": x[7],
        "x8": x[8],     
        "x9": x[9],    
        "x10": x[10],    
        "x11": x[11]                                        
    }
    return d

def extract_extras_from_abspath(abs_path):
    filename = str(abs_path).replace("._","").split(str("/"))[-1]
    return {**{"image": abs_path, **extract_extras_from_filename(filename)}}
