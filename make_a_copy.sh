dst=$1

mkdir ../${dst}
cp -r configs datasets demo_imgs losses models utils \
    *.py *.sh *.txt *.md \
    .flake8 .gitignore .git LICENSE \
    ../${dst}
