import bcrypt

def hash_pw():
    pwd=input("password: ")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(str.encode(pwd), salt)
    print(pwd)
    print(hashed)

if __name__ == "__main__":
    hash_pw()