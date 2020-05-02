python3 -m pip install tensorflow-gpu==1.15
python3 -m pip install tensorflow-datasets
python3 -m pip install glob2 
python3 -m pip install Pillow
python3 -m pip install numpy
python3 -m pip install matplotlib
python3 -m pip install imageio
python3 -m pip install tqdm
python3 -m pip install pathlib

if [ "$1" == "-c" ]; then
	export fileid=1L0sh5pYQbsxFpDRRcTJ5FprBk5CemDc9
	export filename=checkpoints.zip
	wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
	wget --load-cookies cookies.txt -O $filename 'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
	rm -f confirm.txt cookies.txt
	unzip checkpoints.zip
	rm checkpoints.zip
