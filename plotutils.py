import os
import itertools
import io
import codecs
import subprocess;
import numpy as np
import matplotlib.figure as figure
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from PyPDF2 import PdfFileReader, PdfFileWriter


# _base_colors = np.array([(colors.hex2color(x),) for x in ["#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD", "#CCEBC5", "#FFED6F"]])
_base_colors = np.array([(colors.hex2color(x),) for x in ["#8DD3C7", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD", "#CCEBC5", "#FFED6F"]])
# _base_colors = np.array([(colors.hex2color(x),) for x in ["#CAB2D6", "#6A3D9A", "#B2DF8A", "#33A02C", "#A6CEE3", "#1F78B4", "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00", "#6CD9DC", "#43C6C6"]])
# _base_colors = np.array([(colors.hex2color(x),) for x in ["#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A"]])


def getColors(c):
	return np.array([(colors.hex2color(x),) for x in c])[:,0,:]

def getModifiedColorsHSV(c, f):
	mod_colors = colors.rgb_to_hsv(c[:,np.newaxis,:])
	for i in range(mod_colors.shape[0]):
		mod_colors[i,0,:] = f(mod_colors[i,0,0], mod_colors[i,0,1], mod_colors[i,0,2])
	return colors.hsv_to_rgb(mod_colors)[:,0,:]

def getBaseColors():
	return _base_colors[:,0,:]

def getBaseColor(i):
	return _base_colors[i,0,:]
  
# def getBaseColors2():
# 	return _base_colors2[:,0,:]

def getModifiedBaseColorsHSV(f):
	return getModifiedColorsHSV(getBaseColors(), f)

# def getModifiedBaseColors2HSV(f):
# 	return getModifiedColorsHSV(getBaseColors2(), f)


def getBaseStyleCycle():
	return itertools.cycle(iter(["o-", "^-", "s-", "p-", "h-", "x-"]))


def getBoundingBoxes(filename):
	cmd = ["gswin64c" if os.name == "nt" else "gs", "-dBATCH", "-dSAFER", "-dNOPAUSE", "-dQUIET", "-sDEVICE=bbox", filename]
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = p.communicate()
	
	if p.returncode != 0:
		raise Exception(' '.join(cmd) + " failed")
	
	bbs = []
	for l in codecs.decode(err).splitlines():
		s = l.split()
		
		if s and s[0] == "%%HiResBoundingBox:":
			bbs.append((float(s[1]), float(s[2]), float(s[3]), float(s[4])))
	
	return bbs

def cropPDF(filename):
	bbs = getBoundingBoxes(filename)
	
	with io.BytesIO() as out:
		writer = PdfFileWriter()
		
		with open(filename, 'rb') as file:
			reader = PdfFileReader(file)
			
			for i, page in enumerate(reader.pages):
				page.trimBox.lowerLeft = (bbs[i][0], bbs[i][1])
				page.trimBox.upperRight = (bbs[i][2], bbs[i][3])
				page.mediaBox = page.trimBox
				writer.addPage(page)
			
			writer.write(out)
		
		with open(filename, 'wb') as file:
			file.write(out.getbuffer())

def plotLegendToFile(filename, handles, labels, *args, **kwargs):
	leg_fig = figure.Figure(figsize=(128, 128))
	leg_canvas = FigureCanvasPdf(leg_fig)
	leg_ax = leg_fig.add_subplot(1, 1, 1)
	leg_ax.set_axis_off()
	leg = leg_fig.legend(handles, labels, *args, **kwargs)
	leg_canvas.print_figure(filename, bbox_inches='tight')
	cropPDF(filename)


def createFigure(*args, **kwargs):
	fig = figure.Figure(*args, **kwargs)
	canvas = canvas = FigureCanvasPdf(fig)
	return (fig, canvas)


def getDefaultLineStyleCycle():
	return itertools.cycle(iter(["o-", "^-", "s-", "p-", "h-", "x-", "*-", "d-"]))

def getDefaultMarkerCycle():
	return itertools.cycle(iter(["o", "^", "s", "p", "h", "x", "*", "d"]))
