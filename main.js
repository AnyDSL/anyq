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
		this.y_max = d3.max(data, d => this.map_y_data(d));
	}

	map_y_data(d) {
		return NaN;
	}

	update(x, y, line_style_map) {
		const line = d3.line().x(d => x(d.num_threads)).y(d => y(this.map_y_data(d)));

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




class Plot {
	constructor(svg, y_axis, lineType, margin) {
		this.svg = svg;
		this.width = () => svg.width();
		this.height = () => svg.height();
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

	update() {
		this.x_max = d3.max(this.lines, l => l.is_visible() ? l.x_max : this.x_min);
		this.y_max = d3.max(this.lines, l => l.is_visible() ? l.y_max : this.y_min);

		this.x = d3.scaleLog()
				   .base(2)
				   .domain([this.x_min, this.x_max])
				   .range([this.margin.left, this.width() - this.margin.right]);

		this.y = d3.scaleLog()
				   .domain([this.y_min, 1.1 * this.y_max])
				   .range([this.height() - this.margin.bottom, this.margin.top]);


		this.axis.x.call(d3.axisBottom(this.x))
				   .attr("transform", `translate(0, ${this.y(this.y_min)})`);
		this.axis.y.call(d3.axisLeft(this.y).ticks(10, v => { if (!Number.isInteger(Math.log10(v))) return; return `${v}`; }))
				   .attr("transform", `translate(${this.x(1)}, 0)`);
		this.label.x.attr("transform", `translate(${(this.x(this.x_min) + this.x(this.x_max)) / 2}, ${this.y(this.y_min) + 38})`);
		this.label.y.attr("transform", `translate(${this.x(this.x_min) - 34}, ${(this.y(this.y_min) + this.y(this.y_max)) / 2})rotate(-90)`);

		for (let l of this.lines) {
			l.update(this.x, this.y, this.line_style_map);
		}

		return true;
	}
}

class OpsPlot extends Plot {

	update() {
		super.update();

		this.axis.y.call(d3.axisLeft(this.y).ticks(10, v => {
			var e = Math.log10(v);
			if (!Number.isInteger(e)) return;
			if (e >= 6) {
				return `${v/1000000}M`;
			} else if (e >= 3) {
				return `${v/1000}k`;
			}
			return `${v}`;
		}));
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


function fillButtonTable(tableElem, line_map, line_style_map)
{
	let header_row = $("<tr/>").appendTo(tableElem);

	for (const [col_name, entries] of line_map.entries()) {
		$("<th/>").text(col_name).appendTo(header_row);
	}

	let values_row = $("<tr/>").appendTo(tableElem);


	for (const [col_name, entries] of line_map.entries()) {
		let col = $("<td/>")
			.attr("data-category", col_name)
			.appendTo(values_row);

		for (const [entry, lines] of entries) {
			let label = $("<label/>")
				.addClass("checkbox")
				.attr("data-category", col_name)
				.attr("data-value", slugify(String(entry)))
				.appendTo(col);

			let style = line_style_map.get(col_name);

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
			checkbox.appendTo(label);

			$("<span/>").text(entry).appendTo(label);

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
	plot = new Plot(svgElem, "run time/ms", TimingLine, { top: 8, right: 48, bottom: 48, left: 48});
	return plot;
}

function createOpsPlot(svgElem)
{
	class OpsLine extends Line {
		map_y_data(d) {
			// t_avg in nanoseconds
			return 1000000000 / d.t_avg / d.num_threads;
		}
	}

	plot = new OpsPlot(svgElem, "ops per sec per thread", OpsLine, { top: 8, right: 48, bottom: 48, left: 48});
	plot.y_min = 0.001;

	return plot;
}


function createPlot(plotElem)
{
	let svg = $('<svg width="800" height="600" class="plot-graph"></svg>').appendTo(plotElem);
	let menu = $('<div class="plot-menu"></div>').appendTo(plotElem);
	//let share = $('<div class="button-share"><a href="#">Go to preconfigured page</a></div>').appendTo(plotElem);

	let plot;
	if (plotElem.hasClass("plot-time")) {
		plot = createTimePlot(svg);
	} else if (plotElem.hasClass("plot-ops")) {
		plot = createOpsPlot(svg);
	}

	let data = window[plotElem.attr('data-src')];
	//console.log(plotElem, "data:", data);
	if (!data) return;

	const line_map = plot.add_lines(data);

	const line_style_map = new Map();

	for (const [param, values] of line_map) {
		const style = line_styles.get(param);

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

	let button_table = $("<table/>", {
		"class": "button_table"
	});
	button_table.appendTo(menu);
	fillButtonTable(button_table, line_map, line_style_map);

	return plot;
}
