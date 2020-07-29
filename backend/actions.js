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
var nameCol = headersObj["name"];
var typeCol = headersObj["type"];
var addressCol = headersObj["address"];
var messageCol = headersObj["message"];
var additionalCol = headersObj["additional"];

document.getElementById("petA").disabled = document.getElementById("infiniteA").checked;
document.getElementById("petB").disabled = document.getElementById("infiniteB").checked;

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
    row.cells[nameCol].innerHTML = "";
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

function setValue(elementId)	{
	var node = document.getElementById(elementId);
	var value = node.value;
	node.setAttribute("value", value);
}

function setCheckedNode(node)	{
	if (node.checked)	{
		node.setAttribute("checked", "checked");
	}	else	{
		node.removeAttribute("checked");
	}
}

function setChecked(elementId)	{
	setCheckedNode(document.getElementById(elementId));
}

function prepareRow(row)	{
	var cells = row.cells;
	var typeCell = cells[typeCol];
	var children = typeCell.children;
	if (children.length > 0)	{
		var node = children[0];
		//console.log('select node inner html:\n' + node.innerHTML);
		setSelected(node);		
	}
	var additionalCell = cells[additionalCol];
	children = additionalCell.children;
	if (children.length == 2 && children[1].nodeName == "INPUT")	{
		setCheckedNode(children[1]);
	}	
}

function apply()	{
	cycleFunc(prepareRow);
	setValue("PEM");
	setValue("PET");
	for (var elementId of ["infiniteA", "infiniteB", "inverseA", "inverseB"])	{
		setChecked(elementId);
	}
	genNames();
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
	var sendObj = {ids: ids};
	if (ids.includes(1))	{
		var relay1Cell = getRows()[1].cells[additionalCol];
		var relayNode = relay1Cell.children[1];
		if (relayNode.checked)	{
			sendObj["relay1"] = 1;
		} else	{
			sendObj["relay1"] = 0;
		}
	}
	if (ids.includes(2))	{
		var relay2Cell = getRows()[2].cells[additionalCol];
		var relayNode = relay2Cell.children[1];
		if (relayNode.checked)	{
			sendObj["relay2"] = 1;
		} else	{
			sendObj["relay2"] = 0;
		}
	}
	console.log(JSON.stringify(sendObj));
	xhr.send(JSON.stringify(sendObj));
}

function remove()	{
	var table = getTable();
	var rows = getRows();
    for (var i = 1; i < rows.length; i++)	{
    	var row = rows[i];
		actionId = parseInt(row.cells[actionIdCol].innerHTML);
		if (actionId <= 3)	{
			continue;
		}
    	var checkBox = row.cells[checkCol].children[0];
    	if (checkBox.checked == true && table.rows.length > 5)	{
    		table.children[0].removeChild(row);
    	}
    }
}

function genName(phoneNum, name, names)	{
	if (!name)	{
		name = phoneNum.substring(phoneNum.length - 2, phoneNum.length) + "sms";
	}
	if (names.has(name))	{
		var newName;
		for (var i = 2; i < 20; i++) {
			  newName = name + i;
			  if (names.has(newName) == false)	{
				  name = newName;
				  break;
			  }
		}
	}
	return name;
}

function genNames()	{
	var names = new Set();
	var rows = getRows();
	for (var row of Array.from(rows).slice(4))	{
		var cells = row.cells;
	    var name = cells[nameCol].textContent.trim();
	    var phoneNum = cells[addressCol].textContent.trim();
	    name = genName(phoneNum, name, names);
	    cells[nameCol].innerHTML = name;
	    names.add(name);
    }
}

function setInputFilter(textbox, inputFilter) {
	  ["input", "keydown", "keyup", "mousedown", "mouseup", "select", "contextmenu", "drop"]
	  	.forEach(function(event) {
	  				textbox.addEventListener(event, function() {
	  					if (inputFilter(this.value)) {
	  						this.oldValue = this.value;
	  						this.oldSelectionStart = this.selectionStart;
	  						this.oldSelectionEnd = this.selectionEnd;
	  					} else if (this.hasOwnProperty("oldValue") && this.oldValue != "") {
	  						this.value = this.oldValue;
	  						this.setSelectionRange(this.oldSelectionStart, this.oldSelectionEnd);
	  					} else {
	  						this.value = "0";
	  					}
	  				});
	  			});
}

for (var id of ["PEM", "PET", "petA", "petB"])	{
	var element = document.getElementById(id);
	setInputFilter(element, 
				   function(value) { 
				   		return /^\d*$/.test(value) && (value === "" || parseInt(value) < 100); 
				   });
	element.addEventListener("change", function() {
											if (this.value == "")	{
												this.value = 0;
											};
									   });
}

function changeCheck(value, petElement)	{
	if (value)	{
		petElement.disabled = "disabled";
	}	else	{
		petElement.removeAttribute("disabled");
	}
}