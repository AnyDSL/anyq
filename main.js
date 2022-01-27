class LineParams {
	constructor(device, queue_type, queue_size, block_size, p_enq, p_deq, workload_size) {
		this.device = device;
		this.queue_type = queue_type;
		this.queue_size = queue_size;
		this.block_size = block_size;
		this.p_enq = p_enq;
		this.p_deq = p_deq;
		this.workload_size = workload_size;
	}
};

class Result {
	constructor(num_threads, t_avg, t_min, t_max) {
		this.num_threads = num_threads;
		this.t_avg = t_avg;
		this.t_min = t_min;
		this.t_max = t_max;
	}
};

class QueueOperationStatistics {
	constructor(num_operations, t_total, t_min, t_max) {
		this.num_operations = num_operations
		this.t_total = t_total
		this.t_min = t_min
		this.t_max = t_max
	}
};

class QueueOpStatsResult {
	constructor(num_threads, t, n, enqueue_stats_succ, enqueue_stats_fail, dequeue_stats_succ, dequeue_stats_fail) {
		this.num_threads = num_threads;
		this.t = t;
		this.n = n;
		this.enqueue_stats_succ = enqueue_stats_succ;
		this.enqueue_stats_fail = enqueue_stats_fail;
		this.dequeue_stats_succ = dequeue_stats_succ;
		this.dequeue_stats_fail = dequeue_stats_fail;
	}
};

class LineData {
	constructor(params, results) {
		this.params = params;
		this.results = results;
	}
};


class DefaultMap extends Map {
	get(key, gen) {
		let el = super.get(key);

		if (el == undefined) {
			let new_el = gen();
			this.set(key, new_el);
			return new_el;
		}

		return el;
	}
};


class Line {
	constructor(graph, params, data) {
		this.params = params;
		this.data = data;
		this.path = $(graph.append("path").node());
		this.vis = 0;
		this.path.css("visibility", "visible");
		this.x_max = d3.max(data, d => d.num_threads);
		this.y_max = d3.max(data, d => this.defined(d) ? this.map_y_data(d) : 0);
		//this.y_min = d3.min(data, d => this.defined(d) ? this.map_y_data(d) : 0);
	}

	map_y_data(d) {
		return NaN;
	}

	defined(d) {
		return true;
	}

	update(x, y, line_style_map) {
		const line = d3.line().defined(this.defined).x(d => x(d.num_threads)).y(d => y(this.map_y_data(d)));

		this.path.attr("d", line(this.data));

		if (line_style_map) {
			for (const [param, style] of line_style_map) {
				let stylist = style.get(this.params[param]);

				if (stylist) {
					stylist(this.path);
				}
			}
		}
		else {
			this.path.attr("stroke", "black");
		}

		this.stroke_width = this.path.attr("stroke-width");
	}

	is_visible() {
		return this.path.css("visibility") == "visible";
	}

	hide() {
		--this.vis;
		this.path.css("visibility", "hidden");
	}

	show() {
		// lines need to match all previously deselected features to become visible again
		++this.vis;
		if (this.vis == 0)
			this.path.css("visibility", "visible");
	}

	highlight() {
		this.path.attr("stroke-width", 5.0);
	}

	unhighlight() {
		this.path.attr("stroke-width", this.stroke_width);
	}
};

class TimingLine extends Line {
	map_y_data(d) {
		return d.t_avg
	}
}


function formatLog10(v) {
	let e = Math.log10(v);
	if (!Number.isInteger(e)) return;
	if (e == 0) {
		return "1";
	} else if (e == 1) {
		return "10";
	} else {
		return `${v}`;
		//return `10${(e + "").replace(/./g, c => "⁰¹²³⁴⁵⁶⁷⁸⁹"[c] || "⁻")}`;
	}
}

function formatSILog10(v) {
	var e = Math.log10(v);
	if (!Number.isInteger(e)) return;
	if (e >= 6) {
		return `${v/1000000}M`;
	} else if (e >= 3) {
		return `${v/1000}k`;
	}
	return `${v}`;
}

class Plot {
	constructor(svg, y_axis, y_scale, lineType, margin) {
		this.svg = svg;
		this.width = () => 800; //svg.width();
		this.height = () => 600; //svg.height();
		this.margin = margin;
		this.lineType = lineType;

		this.graph = d3.select(this.svg[0]);
		this.axis = {
			x: this.graph.append("g"),
			y: this.graph.append("g")
		};
		this.label = {
			x: this.graph.append("text")
				   .text("number of threads")
				   .attr("text-anchor", "middle")
				   .attr("class", "axis_label"),
			y: this.graph.append("text")
				   .text(y_axis)
				   .attr("text-anchor", "middle")
				   .attr("class", "axis_label")
		};

		this.y_scale = y_scale;
		this.y_tickFormat = null;

		this.x_min = 1;
		this.y_min = 0.0025;
		this.x_max = 0;
		this.y_max = 0;

		this.x = null;
		this.y = null;

		this.lines = [];
	}

	add_lines(data) {
		let line_map = new DefaultMap();

		for (let {params, results} of data) {
			if (results.length == 0) {
				//console.log("skip", params);
				continue;
			}

			let line = new this.lineType(this.graph, params, results);

			for (let key of Object.getOwnPropertyNames(params)) {
				line_map.get(key, () => new DefaultMap()).get(params[key], () => new Set()).add(line);
			}

			this.lines.push(line);
		}

		return line_map;
	}

	update(y_domain) {
		this.x_max = d3.max(this.lines, l => l.is_visible() ? l.x_max : this.x_min);
		this.y_max = d3.max(this.lines, l => l.is_visible() ? l.y_max : this.y_min);

		this.x = d3.scaleLog().base(2)
			.domain([this.x_min, this.x_max])
			.range([this.margin.left, this.width() - this.margin.right]);

		if (y_domain) {
			this.y_min = y_domain[0];
			this.y_max = y_domain[1];
		} else {
			// add 10% padding to the top
			this.y_max *= 1.1;
		}

		this.y = this.y_scale()
			.domain([this.y_min, this.y_max])
			.range([this.height() - this.margin.bottom, this.margin.top]);


		this.axis.x.call(d3.axisBottom(this.x))
				   .attr("transform", `translate(0, ${this.y(this.y_min)})`);
		this.axis.y.call(d3.axisLeft(this.y).ticks(10, this.y_tickFormat))
				   .attr("transform", `translate(${this.x(1)}, 0)`);
		this.label.x.attr("transform", `translate(${(this.x(this.x_min) + this.x(this.x_max)) / 2}, ${this.y(this.y_min) + 38})`);
		this.label.y.attr("transform", `translate(${this.x(this.x_min) - 34}, ${(this.y(this.y_min) + this.y(this.y_max)) / 2})rotate(-90)`);

		for (let l of this.lines) {
			l.update(this.x, this.y, this.line_style_map);
		}

		return true;
	}
}

class RatioPlot extends Plot {

	update() {
		super.update([0, 100]);
	}

}
// const bisect_num_threads = d3.bisector(d => d.num_threads).center;

// let cursor = svg.append("circle")
//                 .attr("r", 2);


// svg.on("mousemove", e => {
// 	// let i = bisect_num_threads(data, x.invert(e.offsetX));
// 	// let num_threads = data[i].num_threads;
// 	// let t = data[i].t;
// 	// cursor.attr("cx", x(num_threads)).attr("cy", y(t));
// 	cursor.attr("cx", e.offsetX).attr("cy", e.offsetY);
// });

function slugify(str) {
	str = str.trim();
	str = str.toLowerCase();

	// remove accents, swap ñ for n, etc
	var from = "åàáãäâèéëêìíïîòóöôùúüûñç·/_,:;";
	var to   = "aaaaaaeeeeiiiioooouuuunc------";

	for (var i = 0, l = from.length; i < l; i++) {
		str = str.replace(new RegExp(from.charAt(i), "g"), to.charAt(i));
	}

	str = str
		.replace(/\.+/g, "-") // replace dots
		.replace(/[^a-z0-9 -]/g, "") // remove invalid chars
		.replace(/\s+/g, "-") // collapse whitespace and replace by -
		.replace(/-+/g, "-") // collapse dashes
		.replace(/^-+/, "") // trim - from start of text
		.replace(/-+$/, ""); // trim - from end of text

	return str;
}

function findOrCreateField(name, parentElem) {
	let elem = parentElem.find('div#pills-' + name);
	if (elem.length > 0)
		return elem;

	let item = $('<li/>')
		.addClass('nav-item')
		.attr('role', 'presentation');
	let btn = $('<button/>')
		.addClass('nav-link')
		.attr('id', 'pills-tab-' + name)
		.attr('data-bs-toggle', 'pill')
		.attr('data-bs-target', '#pills-' + name)
		.attr('type', 'button')
		.attr('role', 'tab')
		.attr('aria-controls', 'pills-' + name)
		.attr('aria-selected', 'false')
		.text(name)
		.appendTo(item);

	let tab = $('<div/>')
		.addClass('tab-pane')
		.attr('id', 'pills-' + name)
		.attr('role', 'tabpanel')
		.attr('aria-labelledby', 'pills-tab-' + name)
		.attr('data-category', name);

	item.appendTo($('#pills-tab'));
	tab.appendTo($('#pills-tabContent'));
	return tab;
}

function findOrCreateLabel(name, entry, style, parentElem) {
	//console.log(parentElem);
	let slug = slugify(String(entry));
	let elem = parentElem.children('label[data-category="' + name + '"][data-value="' + slug + '"]');
	//console.log(elem);
	if (elem.length > 0)
		return elem;

	let label = $("<label/>")
		.addClass("checkbox")
		.attr("data-category", name)
		.attr("data-value", slug)
		.appendTo(parentElem);

	if (style) {
		let stylist = style.get(entry);

		if (stylist) {
			let svg = d3.create("svg")
						.attr("width", 24)
						.attr("height", 9)
						.attr("class", "legend_item");

			let line = svg.append("path")
						  .attr("d", "M 0 4 H 24")
						  .attr("stroke", "black")
						  .attr("stroke-width", 2);

			stylist(line);

			label.append(svg.node());
		}
	}

	let checkbox = $("<input/>", {
		"type": "checkbox",
		"checked": true
	});
	checkbox.appendTo(label);

	$("<span/>").text(entry).appendTo(label);

	return label;
}

function fillButtonTable(menuElem, line_map, line_style_map, plot)
{
	for (const [col_name, entries] of line_map.entries()) {
		let fieldElem = findOrCreateField(col_name, menuElem);

		for (const [entry, lines] of entries) {
			let style = line_style_map.get(col_name);

			let label = findOrCreateLabel(col_name, entry, style, fieldElem);
			//console.log(label);

			let checkbox = label.children('input')

			checkbox.on("change", function() {
				for (let line of lines) {
					//console.log(this, lines);
					if (this.checked) {
						line.show();
						//$.each(lines, function(i, line) { line.show(); });
					} else {
						line.hide();
						//$.each(lines, function(i, line) { line.hide(); });
					}
				}

				plot.update();
			});

			label.mouseenter(e => {
				for (let line of lines) {
					line.highlight();
				}
			});
			label.mouseleave(e => {
				for (let line of lines) {
					line.unhighlight();
				}
			});
		}
	}
}

function createTimePlot(svgElem)
{
	let plot = new Plot(svgElem, "run time (ms)", d3.scaleLog, TimingLine, { top: 8, right: 48, bottom: 48, left: 48});
	plot.y_tickFormat = formatLog10;
	return plot;
}

function createOpsPlot(svgElem)
{
	class OpsLine extends Line {
		map_y_data(d) {
			// t_avg in nanoseconds
			return 1000000000 / d.t_avg / d.num_threads;
		}
		defined(d) {
			return d.t_avg > 0;
		}
	}

	let plot = new Plot(svgElem, "ops/sec per thread", d3.scaleLog, OpsLine, { top: 8, right: 48, bottom: 48, left: 48});
	plot.y_tickFormat = formatSILog10;
	plot.y_min = 0.001;

	return plot;
}

function createThroughputPlot(svgElem)
{
	class ThroughputLine extends Line {
		map_y_data(d) {
			let deq = d.dequeue_stats_succ;
			return 1000.0 * deq.num_operations / d.t;
		}
		defined(d) {
			let deq = d.dequeue_stats_succ.num_operations;
			return Number.isInteger(deq) && deq > 0 && d.t > 0;
		}
	}

	let plot = new Plot(svgElem, "queue elements per sec", d3.scaleLog, ThroughputLine, { top: 8, right: 48, bottom: 48, left: 48});
	plot.y_tickFormat = formatSILog10;
	plot.y_min = 10.0;

	return plot;
}

function createLatencyPlot(svgElem, opField)
{
	class LatencyLine extends Line {
		map_y_data(d) {
			let f = opField(d);
			return 0.001 * f.t_total / f.num_operations;
		}
		defined(d) {
			let f = opField(d);
			//if (!f) return false;
			return f.num_operations > 0 && f.t_total > 0;
		}
	}

	plot = new Plot(svgElem, "avg time / operation (\u00B5s)", d3.scaleLog, LatencyLine, { top: 8, right: 48, bottom: 48, left: 48});
	plot.y_tickFormat = formatLog10;
	plot.y_min = 0.01;

	return plot;
}

function createRatioPlot(svgElem, opField1, opField2)
{
	class RatioLine extends Line {
		map_y_data(d) {
			let f1 = opField1(d);
			let f2 = opField2(d);
			let total = f1.num_operations + f2.num_operations;
			return 100.0 * f1.num_operations / total;
		}
		defined(d) {
			let f1 = opField1(d);
			let f2 = opField2(d);
			return f1.num_operations > 0 || f2.num_operations > 0;
		}
	}

	plot = new RatioPlot(svgElem, "%", d3.scaleLinear, RatioLine, { top: 8, right: 48, bottom: 48, left: 48});

	return plot;
}

function createPlot(plotElem, menuElem, line_style_map)
{
	let svg = $('<svg viewBox="0 0 800 600" class="plot-graph" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"></svg>').appendTo(plotElem);

	let plot;
	if (plotElem.hasClass("plot-time")) {
		plot = createTimePlot(svg);
	} else if (plotElem.hasClass("plot-ops")) {
		plot = createOpsPlot(svg);
	} else if (plotElem.hasClass("plot-throughput")) {
		plot = createThroughputPlot(svg);
	} else if (plotElem.hasClass("plot-latency")) {
		let field;
		if (plotElem.hasClass("plot-latency-enq-success")) {
			field = d => d.enqueue_stats_succ;
		} else if (plotElem.hasClass("plot-latency-enq-failure")) {
			field = d => d.enqueue_stats_fail;
		} else if (plotElem.hasClass("plot-latency-deq-success")) {
			field = d => d.dequeue_stats_succ;
		} else if (plotElem.hasClass("plot-latency-deq-failure")) {
			field = d => d.dequeue_stats_fail;
		}
		if (!field) return;

		plot = createLatencyPlot(svg, field);
	} else if (plotElem.hasClass("plot-percent")) {
		let f1, f2;
		if (plotElem.hasClass("plot-ratio-enq-success")) {
			f1 = d => d.enqueue_stats_succ;
			f2 = d => d.enqueue_stats_fail;
		} else if (plotElem.hasClass("plot-ratio-deq-success")) {
			f1 = d => d.dequeue_stats_succ;
			f2 = d => d.dequeue_stats_fail;
		}
		if (!f1 || !f2) return;

		plot = createRatioPlot(svg, f1, f2);
	}
	if (!plot) return;

	let data = window[plotElem.attr('data-src')];
	//console.log(plotElem, "data:", data);
	if (!data) return;

	const line_map = plot.add_lines(data);


	for (const [param, values] of line_map) {
		let style = line_style_map.get(param);
		if (style)
			continue;

		style = line_styles.get(param);
		if (style) {
			const style_map = new Map();

			let i = 0;
			for (const value of values.keys()) {
				let style_attr = style.attr;
				let style_value = style.values[i % style.values.length];

				style_map.set(value, path => {
					path.attr(style_attr, style_value);
					path.attr("fill", "none");
				});

				++i;
			}

			line_style_map.set(param, style_map);
		}
	}

	plot.line_style_map = line_style_map;
	plot.update();

	fillButtonTable(menuElem, line_map, line_style_map, plot);

	let saveBtn = $('<div class="plot-save-btn d-flex justify-content-end"><a class="btn btn-outline-secondary">Download</a></div>')
		.appendTo(plotElem)
		.find('.btn').click(function(e) {
			let active = $("label.checkbox")
				.filter(function(idx, label) {
					return $(label).find("input[type='checkbox']").prop("checked");
				})
				.map(function () {
					let label = $(this);
					return label.attr("data-category") + ':' + label.attr("data-value");
				})
				.get().join(', ');

			let blob = new Blob([
					'<?xml version="1.0" encoding="UTF-8" standalone="no"?>', '\n',
					'<!-- ', active, ' -->', '\n',
					svg.prop('outerHTML')
				], {type: 'image/svg+xml'});

			let a = $(this);
			a.attr('href', window.URL.createObjectURL(blob));
			a.attr('download', 'plot.svg');
		});

	return plot;
}
