import bcrypt

USERS = {"admin": "Ilya2025", "juan": "Juan2025"}  # cambia contraseñas si quieres

for user, pwd in USERS.items():
    print(user, bcrypt.hashpw(pwd.encode(), bcrypt.gensalt(12)).decode())