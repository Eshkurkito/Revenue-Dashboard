import bcrypt

USERS = {"Marta": "Marta2025"}  # cambia contraseñas si quieres

for user, pwd in USERS.items():
    print(user, bcrypt.hashpw(pwd.encode(), bcrypt.gensalt(12)).decode())