var headersObj = new Object();
{	
	var row = document.getElementById("rulesTable").rows[0];
    var headerCells = row.children;
    for (var i = 0; i < headerCells.length; i++)  {
        var header = headerCells[i].innerHTML;
        headersObj[header] = i;
    }
}

var triggersDic;

var actionsDic;

var checkCol = headersObj["check"];
var ruleIdCol = headersObj["rule_id"];
var formulaCol = headersObj["formula"];
var ruleValCol = headersObj["val"];
var actionCol = headersObj["actions"];

var timerVar = setInterval(updateFunc, 1000);

var triggersObj = {"1": 0, "2": 0, "4": 0};

var rulesObj = getRulesObj();

var sessionId = Math.floor(Math.random() * 1000000) + 1;

initFunc();

function updateTriggers (triggersObj, ruleCell) {
	var children = ruleCell.children;
	for (var i = 0; i < children.length; i++)	{
		var node = children[i];
		//console.log('nodeName:' + node.nodeName);
		if (node.nodeName == "IMG") 	{
			var triggerNode = children[i - 1];
			var options = triggerNode.options;
			var selectedIndex = triggerNode.selectedIndex;
			var triggerName = options[selectedIndex].text;
			//console.log('triggerIdStr:' + triggerIdStr);
			var triggerIdStr;
			for (triggerIdStr in triggersDic)	{
				if (triggerName == triggersDic[triggerIdStr])	{
					break;
				}
			}
			var triggerVal = triggersObj[triggerIdStr];
			//console.log('triggerVal:' + triggerVal);
			var src;
			if (triggerVal)	{
				src = "img\\green16.jpg";
			}	else	{
				src = "img\\gray16.jpg";
			}
			node.setAttribute("src", src);
		} 
	}
}

function updateRules(rulesObj)	{
    var table = document.getElementById("rulesTable");
    var rows = table.rows;
    for (var i = 1; i < rows.length; i++)	{
    	var row = rows[i];
    	var imgNode = row.cells[ruleValCol].children[0];
      	var ruleId = row.cells[ruleIdCol].innerHTML;
      	var src;
       	//console.log('ruleId:' + ruleId + ' ruleVal:' + rulesObj[ruleId]);
      	if (ruleId in rulesObj)	{
      		if (rulesObj[ruleId])	{
      			src = "img\\circle-green.jpg";
      		}	else	{
      			src = "img\\circle-gray.jpg";
      		}
      		imgNode.setAttribute("src", src);
      	}
    }	
}

function updateFunc () {
	var xhr = new XMLHttpRequest();
	xhr.open("POST", "rule", true);
	xhr.setRequestHeader("Content-Type", "application/json");
	xhr.onreadystatechange = function () {
		if (xhr.readyState === 4 && xhr.status === 200) {
		    //console.log('response:' + xhr.responseText);
			respObj = JSON.parse(xhr.responseText);
			triggersObj = respObj['triggers'];
			rulesObj = respObj['rules'];
			//console.log('rulesObj from server:' + JSON.stringify(rulesObj));
			updateRules(rulesObj);
			var rows = document.getElementById("rulesTable").rows;
			for (var i = 1; i < rows.length; i = i + 1)	{
				var row = rows[i];
				var ruleCell = row.cells[formulaCol];
				updateTriggers(triggersObj, ruleCell);
			}
		}
	};
	var rulesObj = getRulesObj();
	//console.log('rulesObj:' + JSON.stringify(rulesObj));
	var data = {triggers: triggersObj, sessionId: sessionId, rules: rulesObj};
	xhr.send(JSON.stringify(data));
}

function initFunc () {
	var xhr = new XMLHttpRequest();
	xhr.open("POST", "initRule", true);
	xhr.setRequestHeader("Content-Type", "application/json");
	xhr.onreadystatechange = function () {
		if (xhr.readyState === 4 && xhr.status === 200) {
		    //console.log('response:' + xhr.responseText);
			var responseObj = JSON.parse(xhr.responseText);
			triggersDic = responseObj['triggers'];
			actionsDic = responseObj['actions'];
			console.log("actions dic:" + JSON.stringify(actionsDic));
			var triggersIds = [];
			for (triggerId in triggersDic)	{
				triggersIds.push(triggerId);
			}
			triggersIds.sort();
			var triggerNames = [];
			for (var triggerId of triggersIds)	{
				triggersObj[triggerId] = 0;
				triggerNames.push(triggersDic[triggerId]);
			}
			
			var actionIds = [];
			for (actionId in actionsDic)	{
				actionIds.push(actionId);
			}
			actionIds.sort();

			var rows = document.getElementById("rulesTable").rows;
			var triggerNode = rows[1].cells[formulaCol].children[0];
			var triggerOptions = triggerNode.children;
			var actionOptions = rows[1].cells[actionCol].children[0].options;
			var prevTriggerIds = [];
			var prevActionIds = [];
			for (var option of triggerOptions)	{
				prevTriggerIds.push(option.getAttribute("trigger_id"));
			}
			for (var option of actionOptions)	{
				prevActionIds.push(option.getAttribute("action_id"));
			}
			rows[0].cells[ruleIdCol].style.display = "none";
			for (var i = 1; i < rows.length; i = i + 1)	{
				var row = rows[i];
				row.cells[ruleIdCol].style.display = "none";
				var ruleCell = row.cells[formulaCol];
				fillTriggers(ruleCell, prevTriggerIds);
				var actionCell = row.cells[actionCol];
				fillActions(actionCell, prevActionIds);
			}
		}
	}
	console.log('triggers obj:' + triggersObj);
	xhr.send();
}

function fillTriggers (ruleCell, prevIds)	{
	for (var triggerNode of ruleCell.children)	{
		if (triggerNode.nodeName != "SELECT" || 
				!triggerNode.children[0].hasAttribute("trigger_id"))	{
			continue;
		}
		for (var option of triggerNode.options)	{
			var triggerName = option.innerHTML;
			var triggerId = option.getAttribute("trigger_id");
			var selected = option.hasAttribute("selected");
			if (triggerId in triggersDic)	{
				option.innerHTML = triggersDic[triggerId];
				continue;
			}
			if (!selected)	{
				triggerNode.removeChild(option);
				continue
			}
			for (triggerId in triggersDic)	{
				if (triggersDic[triggerId] == triggerName)	{
					option.setAttribute("trigger_id", triggerId);
				}
			}
		}
		for (var triggerId in triggersDic)	{
			if (!prevIds.includes(triggerId))	{
				console.log("prevIds " + prevIds + " not includes " + triggerId);
				var option = document.createElement("option");
				option.setAttribute("trigger_id", triggerId);
				option.textContent = triggersDic[triggerId];
				triggerNode.appendChild(option);
			}
		}
	}
}

function fillActions (actionCell, prevIds)	{
	for (var actionNode of actionCell.children)	{
		if (actionNode.nodeName != "SELECT")	{
			continue;
		}
		console.log("actions dic:" + JSON.stringify(actionsDic));
		for (var option of actionNode.options)	{
			var actionName = option.innerHTML;
			var actionId = option.getAttribute("action_id");
			var selected = option.hasAttribute("selected");
			console.log("name " + actionName + " selected:" + selected);
			if (actionId in actionsDic)	{
				console.log("action id " + actionId + " in actionsDic, action name " + actionName);
				option.innerHTML = actionsDic[actionId];
				continue;
			}
			if (!selected)	{
				console.log("remove option");
				actionNode.removeChild(option);
				continue
			}
			for (actionId in actionsDic)	{
				if (actionsDic[actionId] == actionName)	{
					option.setAttribute("action_id", actionId);
				}
			}
		}
		for (var actionId in actionsDic)	{
			if (!prevIds.includes(actionId))	{
				var option = document.createElement("option");
				option.setAttribute("action_id", actionId);
				option.textContent = actionsDic[actionId];
				actionNode.appendChild(option);
			}
		}
	}
}

function addRule() {
    var table = document.getElementById("rulesTable");
    var rows = table.rows;
    var len = rows.length
    var row = rows[len - 1].cloneNode(true);
    var ind = parseInt(row.cells[ruleIdCol].innerHTML) + 1;
    row.cells[ruleIdCol].innerHTML = ind;
    table.children[0].appendChild(row);
}

function setSelected(node) 	{
	if (node.nodeName == "SELECT")	{
		var html1 = node.innerHTML;
		var options = node.children;
		var len = options.length;
		var selectedIndex = node.selectedIndex;
		console.log('selected index:' + selectedIndex);
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

function apply()	{
    var table = document.getElementById("rulesTable");
    var rows = table.rows;
    for (var i = 1; i < rows.length; i++)	{
    	var row = rows[i];
    	var ruleCell = row.cells[formulaCol];
    	var children = ruleCell.children;
    	for (var j = 0; j < children.length; j++)	{
    		var node = children[j];
       		setSelected(node);
    	}
    	var actionCell = row.cells[actionCol];
    	children = actionCell.children;
    	for (var actionNode of children)	{
    		setSelected(actionNode);
    	}
    }
	var xhr = new XMLHttpRequest();
	var url = "applyRules";
	xhr.open("POST", url, true);
	xhr.setRequestHeader("Content-Type", "application/html");

	var pageHTML = "<html>\n" + document.documentElement.innerHTML + "\n</html>";
	var data = JSON.stringify({"html": pageHTML, "sessionId":  sessionId});
	xhr.send(data);
	setTimeout(nullifyVals, 3000);
}

function nullifyVals()	{
	//console.log('time out');
	var rows = document.getElementById("rulesTable").rows;
	for (var i = 1; i < rows.length; i++) {
		var imgNode = rows[i].cells[ruleValCol].children[0];
		imgNode.setAttribute("src", "img\\circle-gray.jpg");
	}
}

function getRulesObj()	{
	var rulesObj = {};
    var table = document.getElementById("rulesTable");
    var rows = table.rows;
    for (var i = 1; i < rows.length; i++)	{
    	var row = rows[i];
    	var ruleId = parseInt(row.cells[ruleIdCol].innerHTML);
	    var src = row.cells[ruleValCol].children[0].getAttribute("src");
	    var ruleVal;
	    if (src == "img\\circle-green.jpg")	{
	    	ruleVal = 1;
	    }	else	{
	    	ruleVal = 0;
	    }
    	rulesObj[ruleId] = ruleVal;
    }
    return rulesObj;
}

function remove()	{
	var table = document.getElementById("rulesTable");
	var rows = table.rows;
	for (var row of Array.from(rows).slice(1))	{
    	var checkBox = row.cells[checkCol].children[0];
    	if (checkBox.checked == true && rows.length > 2)	{
	    		table.children[0].removeChild(row);
    	}
    }
}

function addTrigger(refNode)	{
	var imgNode = refNode.previousElementSibling;
	var triggerNode = imgNode.previousElementSibling;
	var imgNode = imgNode.cloneNode(true);
	var triggerNode = triggerNode.cloneNode(true);
	
	var triggersCell = refNode.parentNode;
	var nodes = triggersCell.children;
	var node;
	var curNames = [];
	var curTriggerNode;
	for (node of nodes)	{
		if (node.nodeName == "IMG")	{
			curTriggerNode = node.previousElementSibling;
			curNames.push(curTriggerNode.value);
		}
	}
	var nOfTriggers = curNames.length; 
	if (nOfTriggers > 7)	{
		return;
	}
	if (nOfTriggers == 4)	{
		node = document.createElement("small");
		node.innerHTML = ">>";
		triggersCell.insertBefore(node, refNode);
		node = node.cloneNode(true);
		triggersCell.insertBefore(node, refNode);
		var brNode = document.createElement("br");
		triggersCell.insertBefore(brNode, node);
	}

	var selected = false;
	for (var option of triggerNode.options)	{
		if (curNames.includes(option.innerHTML) || selected)	{
			option.removeAttribute("selected");
		}	else	{
			option.setAttribute("selected", "selected");
			selected = true;
		}
	}
	var opNode = document.createElement('select');
	opNode.setAttribute("class", "operation");
	for (var op of ["and", "and not", "or", "or not"])	{
		var subNode = document.createElement("option");
		subNode.innerHTML = op;
		opNode.appendChild(subNode);
	}
	opNode.children[0].setAttribute("selected", "selected");
	triggersCell.insertBefore(opNode, refNode);
	triggersCell.insertBefore(triggerNode, refNode);
	triggersCell.insertBefore(imgNode, refNode);
}

function removeTrigger(triggersCell)	{
	var nodes = triggersCell.children;
	var len = nodes.length;
	var i = len - 1;
	//console.log("nodes[i].name:" + nodes[i].name);
	while (nodes[i].nodeName != "IMG")	{
		i = i - 1;
	}
	if (i < 3)	{
		return;
	}
	while (true)	{
		triggersCell.removeChild(nodes[i]);
		i = i - 1;
		if (nodes[i].nodeName == "IMG")	{
			break;
		} 
	}
}

function addAction(refNode)	{
	var actionCell = refNode.parentNode;
	var nodes = actionCell.children;
	var len = nodes.length;
	var i = len - 1;
	//console.log("nodes[i].name:" + nodes[i].name);
	while (nodes[i].nodeName != "SELECT")	{
		i = i - 1;
	}
	if (i > 2)	{
		return;
	}
	var actionNode = nodes[i].cloneNode(true);
	var curActions = [];
	for (var j = 0; j <= i; j++)	{
		curActions.push(nodes[j].value);
	}
	var selected = false;
	for (var option of actionNode.options)	{
		if (curActions.includes(option.innerHTML) || selected)	{
			option.removeAttribute("selected");
		}	else	{
			option.setAttribute("selected", "selected");
			selected = true;
		}
	}
	actionCell.insertBefore(actionNode, refNode);
}

function removeAction(actionCell)	{
	var nodes = actionCell.children;
	var len = nodes.length;
	var i = len - 1;
	//console.log("nodes[i].name:" + nodes[i].name);
	while (nodes[i].nodeName != "SELECT")	{
		i = i - 1;
	}
	if (i < 1)	{
		return;
	}
	actionCell.removeChild(nodes[i]);
}
