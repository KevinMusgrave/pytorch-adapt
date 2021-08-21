rm -rfv build/ && \
rm -rfv dist/ && \
rm -rfv src/pytorch_adapt.egg-info/ && \
python3 setup.py sdist bdist_wheel