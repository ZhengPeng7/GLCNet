dst='GLCNet_codes'

mkdir ../${dst}
cp -r configs datasets demo_imgs models utils \
    *.py *.sh *.txt *.md \
    .flake8 .gitignore .git LICENSE \
    ../${dst}

