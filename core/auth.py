import re

def name_valid(name):
    if name.isalpha() and len(name) > 1:
        return True
    else:
        return false


def password_valid(pass1):
    reg = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!#%*?&]{6,20}$"

    pat = re.compile(reg)
    mat = re.search(pat, pass1)

    if mat and len(pass1) >= 6:
        return True
    else:
        return False

def mobile_valid(mobile):
    if len(mobile) == 10:
        return True
    else: 
        return False

#check password
def password_check(pass1, pass2):
    if pass1 == pass2:
        return True
    else:
        return False

def authentication(first_name, last_name, mobile, pass1, pass2):
    if name_valid(first_name) == False:
        return "Invalid First Name"
    elif name_valid(last_name) == False:
        return "Invalid Last Name"
    elif mobile_valid(mobile) == False:
        return "Invalid Mobile Number"
    elif password_valid(pass1) == False:
        return "Password Should be in Proper Format. (eg. Password@1234)"
    elif password_check(pass1, pass2) == False:
        return "Password Not Matched"
    else:
        return "success"