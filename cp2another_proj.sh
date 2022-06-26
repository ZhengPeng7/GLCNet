src='0_test_best'
dst='GLCNet_codes'

mkdir ../${dst}
cp -r configs datasets models utils *.py *.sh *.txt *.md .flake8 .gitignore .git ../${dst}

mkdir ../${dst}/exp_cuhk
cp exp_cuhk/epoch_17.pth ../${dst}/exp_cuhk
mkdir ../${dst}/exp_prw
cp exp_prw/epoch_17.pth ../${dst}/exp_prw
mkdir ../${dst}/exp_mvn
cp exp_prw/epoch_19.pth ../${dst}/exp_prw
