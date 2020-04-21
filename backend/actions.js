function getTable()	{
	var table = document.getElementById("actionTable");
	return table;
}

function getRows()	{
	var table = getTable();
	var rows = table.rows;
	return rows;
}

var headersObj = new Object();
{	
	var row = getRows()[0];
    var headerCells = row.children;
    for (var i = 0; i < headerCells.length; i++)  {
        var header = headerCells[i].innerHTML;
        headersObj[header] = i;
    }
}
//console.log('headersObj:' + JSON.stringify(headersObj));
var checkCol = headersObj["check"];
var actionIdCol = headersObj["action_id"];
var typeCol = headersObj["type"];
var addressCol = headersObj["address"];
var messageCol = headersObj["message"];

function cycleFunc(f)	{
	var retVal = [];
	var rows = getRows();
    for (var i = 1; i < rows.length; i++)	{
    	var row = rows[i];
    	var val = f(row);
    	retVal.push(val);
    }
    return retVal;
}
  
function add() {
	var table = getTable();
    var rows = getRows();
    var len = rows.length;
    var row = rows[len - 1].cloneNode(true);
    var ind = parseInt(row.cells[actionIdCol].innerHTML) + 1;
    row.cells[actionIdCol].innerHTML = ind;
    table.children[0].appendChild(row);
}

function setSelected(node) 	{
	if (node.nodeName == "SELECT")	{
		var options = node.children;
		var len = options.length;
		var selectedIndex = node.selectedIndex;
		for (var i = 0; i < options.length; i++)	{
			var option = options[i];
			if (i == selectedIndex)	{
				option.setAttribute("selected", "selected");
			} else	{
				option.removeAttribute("selected");
			}
		}
	}	
}

function prepareRow(row)	{
	var cells = row.cells;
	var typeCell = cells[typeCol];
	var children = typeCell.children;
	if (children.length > 0)	{
		var node = children[0];
		console.log('select node inner html:\n' + node.innerHTML);
		setSelected(node);		
	}
}

function apply()	{
	cycleFunc(prepareRow);
	var xhr = new XMLHttpRequest();
	var url = "applyActions";
	xhr.open("POST", url, true);
	xhr.setRequestHeader("Content-Type", "application/html");
	var pageHTML = "<html>\n" + document.documentElement.innerHTML + "\n</html>";
	xhr.send(pageHTML);
}

function getId(row)	{
	var checkBox = row.cells[checkCol].children[0];
	var actionId = 0;
	if (checkBox.checked == true)	{
		var cell = row.cells[actionIdCol];
		actionId = parseInt(cell.innerHTML);
	}
	return actionId;
}

function test()	{
	var ids = cycleFunc(getId);
	function isPositive(value) {
		return value > 0;
	}
	ids = ids.filter(isPositive);
	var xhr = new XMLHttpRequest();
	var url = "testActions";
	xhr.open("POST", url, true);
	xhr.setRequestHeader("Content-Type", "application/html");
	xhr.send(JSON.stringify(ids));
}