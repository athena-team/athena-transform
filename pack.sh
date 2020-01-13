
#1. Uninstall transform
echo "Uninstall transform if exist ..."
pip uninstall -y transform

#2. gen whl file
echo "pack ..."
rm -rf build/  transform.egg-info/ dist/
python3 setup.py bdist_wheel sdist
