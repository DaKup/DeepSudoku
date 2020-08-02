import re, sys, os
import matplotlib.pyplot as plt

from utils.functions import MovingAverage
import json

with open(sys.argv[1], 'r') as f:
	inp = f.read()

patterns = {
	'train': re.compile(r'\[\s*(?P<epoch>\d+)\]\s*(?P<iter>\d+) \|\| B: (?P<b>\S+) \| C: (?P<c>\S+) \| M: (?P<m>\S+) \|( S: (?P<s>\S+) \|)? T: (?P<t>\S+)'),
	'val': re.compile(r'\s*(?P<type>[a-z]+) \|\s*(?P<all>\S+)')
}
data = {key: [] for key in patterns}

for line in inp.split('\n'):
	for key, pattern in patterns.items():
		f = pattern.search(line)

		test = json.loads(line)
		datum = test['data']

		if 'autoscale' in datum.keys():
			continue

		if 'box' not in datum.keys():
			continue
		
		# if f is not None:
			# datum = f.groupdict()
		for k, v in datum.items():
			if v is not None:
				try:
					v = float(v)
				except:
					pass
				datum[k] = v
		
		if key == 'val':
			continue
			datum = (datum, data['train'][-1])
		data[key].append(datum)
		break


def smoother(y, interval=100):
	avg = MovingAverage(interval)

	for i in range(len(y)):
		avg.append(y[i])
		y[i] = avg.get_avg()
	
	return y

def plot_train(data):
	plt.title(os.path.basename(sys.argv[1]) + ' Training Loss')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')

	loss_names = ['BBox Loss', 'Conf Loss', 'Mask Loss']

	x = [x['iter'] for x in data]
	plt.plot(x, smoother([y['loss']['B'] for y in data]))
	plt.plot(x, smoother([y['loss']['C'] for y in data]))
	plt.plot(x, smoother([y['loss']['M'] for y in data]))

	if data[0]['loss']['S'] is not None:
		plt.plot(x, smoother([y['loss']['S'] for y in data]))
		loss_names.append('Segmentation Loss')

	plt.legend(loss_names)
	plt.show()

def plot_val(data):
	plt.title(os.path.basename(sys.argv[1]) + ' Validation mAP')
	plt.xlabel('Epoch')
	plt.ylabel('mAP')

	x = [x['epoch'] for x in data]# if x[0]['type'] == 'box']
	plt.plot(x, [x['box']['all'] for x in data])# if x[0]['type'] == 'box'])
	plt.plot(x, [x['mask']['all'] for x in data])# if x[0]['type'] == 'mask'])

	plt.legend(['BBox mAP', 'Mask mAP'])
	plt.show()

if len(sys.argv) > 2 and sys.argv[2] == 'val':
	plot_val(data['train'])
else:
	plot_train(data['train'])
