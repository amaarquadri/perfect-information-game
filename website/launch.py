import os


def main():
    os.system('cmd /k "cd frontend && npm run dev && cd ../ && python manage.py runserver"')


if __name__ == '__main__':
    main()
