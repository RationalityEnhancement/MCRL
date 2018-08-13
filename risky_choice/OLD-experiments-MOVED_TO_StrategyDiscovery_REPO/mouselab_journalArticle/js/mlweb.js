 /*
 	MouselabWEB  
                
     this script contains functions to display MouselabWEB content
 								
       v 1.00b2, May  14, 2010
	
		improvement from v1.00b: updated reorder function to fixed the problem that the overlay 
		images did not resize with the counterbalancing
	   
	   (c) 2003-2010 Martijn C. Willemsen and Eric J. Johnson 

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

dtNewDate = new Date(); 
starttime = dtNewDate.getTime(); // abs. starttime of experiment

// set vars for delay 
prevtime = 0; 	// memory in timefunction to compensate for delay
dtime=0;		 
prevCell = -1; 	// delay lag memory (-1 means first cell is not delayable 
loaded = false; // flag to test whether page has been loaded 9is set to true in reorder
boxOpen = false; // flag to test whether box is already open (using in showCont and hideCont)
chkFrm = false // flag to test whether additional form elements have to be checked on submission
warningTxt = "Some questions have not been answered. Please answer all questions before continuing!";
CBpreset = false; // flag to set CB order fixed by a matrix on forehand: default is set to false 
prevfieldname = ""; // default for switching clicks

transpImg = new Image()
tempImg = new Image()   

previousSrc="";

// default values
mlweb_outtype="XML";
mlweb_fname=0;
masterCond = 1;  // number of master conditions
randomOrder = false; // force randomize of counterbalancing order
subject="";
evtOpen = 0;
evtClose = 0;

tmDirectStart = false;
tmTimeUp = false;
tmActive = false;
tmCurTime = 0;

// get source transparant image
transpImg.src="transp.gif"

function abc_num(str)
{
out=str.toUpperCase().charCodeAt(0)-65;
return out
}

function fac(x)
{
// Faculty: x!=x(x-1)...1
var outp=1;
for (var i=1; i<=x; i++)
{outp=outp*i}
return outp
}

function CountBal(subjnr, num)
{
// counterbalance based on subj number. 
// first subject is 0
// Num is number of options to counterbalance
// (number of orders is Num!)

var numOrd=fac(num);
start = subjnr - numOrd*Math.floor((subjnr-1)/numOrd)

orderstr=""
for (var i=0;i<num;i++)
{orderstr+=i.toString()}

outstr=""
for (var i=num; i>0; i--)
{
var den=fac(i-1);
pos = Math.floor((start-1)/den)+1
outstr+=orderstr.charAt(pos-1)+","
orderstr = orderstr.substring(0,pos-1)+orderstr.substr(pos)
start=start-(pos-1)*den
}
outstr=outstr.substr(0,outstr.length-1)
return outstr.split(",")
}

function ExpMatrix(M)
{ // expand data matrices
var Mrows=M.split("`");

var outM = new Array();
for (rowcount=0;rowcount<Mrows.length;rowcount++)
	{
	outM[rowcount]=Mrows[rowcount].split("^")
	}
return outM;
}

function ExpRow(M)
{ // expand data vectors

var outM = new Array();
outM = M.split("^") 
return outM;
}

function btnHover(loc,act)
{
	if (act=='out') 
		{
			if (loc.className.indexOf(" ")>0) {tempstyle= loc.className.substring(0,loc.className.indexOf(" "));}
		} 
		else 
		{tempstyle=loc.className + ' btnhov';}

	loc.className=tempstyle;
}

function timefunction(event,name,value) {
// Record proc data in form element
mlweb_form=document.forms[mlweb_fname].elements['procdata']

	dtNewDate = new Date();
	eventtime = dtNewDate.getTime();
  	var curtime = eventtime-starttime-dtime;  // dtime is to compensate for delay time (failed openings have negative time!
//	if (prevtime>curtime) {curtime=prevtime;} else {prevtime=curtime}; // check with previous event time: if smaller, then delay was not finished: set curtime to prevtime so event has duration 0;

	dtime=0; // reset dtime
  	if (mlweb_outtype=="XML")
	   	{
	   	var str="<eventblock><event>"+event+"</event><name>"+name+"</name><value>"+value+"</value><time>"+curtime+"</time></eventblock>";
  		var headerstr="<?xml version=1.0?>"
		}
		else 
		{
		var str="\""+event+"\",\""+name+"\",\""+value+"\",\""+curtime+"\"\n"
		var headerstr="\"event\",\"name\",\"value\",\"time\"\n"
		};
		
	if(mlweb_form.value=='') 
		{
		mlweb_form.value=headerstr;
  		}
 	mlweb_form.value+=str;

if (event=="onload") {reorder();}
return true;
}

// convert event to eventdata and call save function
function RecordEventData(objActionElement, objEvent)
	{
	var strName, strEventType, strFormValue;
	strName = objActionElement.name;
	strFormValue = (objActionElement.value) ? objActionElement.value : "";
	strEventType = objEvent.type;

	//call timefunction 
	timefunction(strEventType,strName, strFormValue)
	return false;
	}
	
function checkForm(formHandle)
{
if (chkFrm) 
	{
	noElm = document.forms[mlweb_fname].elements.length;

	var filled=true;

	for (i=0;i<noElm;i++)
		{	
		elemHandle = document.forms[0].elements[i];
		if (elemHandle.type=="hidden") {continue}; 
		if (elemHandle.value=="") {filled = false; break};
		if (elemHandle.type=="select-one") {if (elemHandle.options[elemHandle.selectedIndex].value=="") {filled = false; break};}
		if (elemHandle.type=="radio")   // procedure to check radio buttons
		   { 
	   		 radio_name=elemHandle.name;  // get name (needed to retrieve length)
		 							  // get length of radio button group	   
	   	 	r_length = eval("document.forms[0]."+radio_name).length

		 	for (ri=0;ri<r_length;ri++)  // check each button and break loop if checked button was found
		 	{ radioHandle = document.forms[0].elements[i+ri];
		 	  if (radioHandle.checked) {filled=true; break} else {filled=false};
			}
			if (filled) {i=i+r_length-1; continue} else {break};  // if checked button found; continue
		   							 		   				  // else break loop and show warning
			}
		
	}
if (!filled) {alert(warningTxt);timefunction('submit','submit','failed');return false};
}

if ((chkchoice=="nobuttons") | !chkFrm) {return true;}
if (chkchoice==true) {timefunction('submit','submit','succeeded');return true} else {alert(warningTxt);timefunction('submit','submit','failed');return false};
}


function objElem(name,value)
{
this.name=name
this.value=value
}

function ShowCont(fieldname, objEvent)
{
if (!loaded) {return;} // do not open boxes when page is loading
if (!tmDirectStart & tmActive & tmCurTime==0) {startTmBar();}
if (tmTimeUp) {return;}
if (hasGambled) {return;}
// check if a click on a link (A) occurs. this happens for example 
// when a mlweb A link gets a focus (due to clicking the box) and a subject presses enter
// this is to prevent enters from generating events when in click (rather than mouseover) mode

if (objEvent.srcElement)
	{
		if (objEvent.srcElement.nodeName=="A") {return}
	}
	else if (objEvent.target)
	{
		if (objEvent.target.nodeName=="A") {return}
	}
	
var row = abc_num(fieldname);
var col = parseInt(fieldname.substr(1));

thisElem = new objElem
// check if open cell should be recorded
if ((statecont[RowOut[row]][ColOut[col]]=="0") & !(recOpenCells)) {return;}
if (boxOpen) {return;}

if (evtClose<3) {boxOpen = true;} //set flag to show box is open

// retrieve tagname and txt for this cell
thisElem.name = tagcont[RowOut[row]][ColOut[col]];
thisElem.value = txtcont[RowOut[row]][ColOut[col]];

RecordEventData(thisElem, objEvent);
nr_acquisitions_by_problem_type[block_nr-1][problem_type[block_nr-1][trial_nr-1]-1]++;
nr_acquisitions[block_nr-1][trial_nr-1]++;

    acquisition={
    	gamble:col,
    	outcome:row+1
    }
    
    acquisitions[block_nr-1][trial_nr-1].push(acquisition);
    
if (document.getElementById)  
	{
	// IE6/NS6>/Mozilla
	HandleTxt = document.getElementById(fieldname+"_txt");
	HandleBox = document.getElementById(fieldname+"_box");
	}
	else if (document.all)
	{
	//IE4/5
	HandleTxt=eval("document.all['"+fieldname+"_txt"+"']");
	HandleBox=eval("document.all['"+fieldname+"_box"+"']");
	}

// delay

currCell = -1; 
for (var i=0;i<Dlist.length;i++)
	{if (tagcont[RowOut[row]][ColOut[col]]==Dlist[i]) {currCell=i;break;}}

if ((prevCell!=-1)&(currCell!=-1)) {dtime = DTimes[currCell][prevCell];} else {dtime=0}; 
prevCell = currCell;

//HandleTxt.style.visibility='visible';HandleBox.style.visibility='hidden';

delay=window.setTimeout("HandleTxt.style.visibility='visible';HandleBox.style.visibility='hidden';",dtime)  //make image transparant

for (r=0; r<matrices.names.length;r++){ 
    for (c=1;c<matrices.names[0].length;c++){
        if (matrices.names[r][c]==fieldname){
            matrices.newly_revealed = [r, c-1];
        }
    }
}

matrices = update_pseudorewards(matrices,decision_problem);

}

function HideCont(fieldname,objEvent)
{
if (!loaded) {return;} // do not open boxes when page is loading
if (!boxOpen) {return;} // do not close boxes that are not open...
//if (click_nr==nr_clicks-1) {$("#NrClicks").html(0); return;}
//click_nr++
//$("#NrClicks").html(nr_clicks-click_nr)

window.clearTimeout(delay);

var row = abc_num(fieldname);
var col = parseInt(fieldname.substr(1));

// check if open cell should be recorded
if ((statecont[RowOut[row]][ColOut[col]]=="0") & !(recOpenCells)) {return;}

boxOpen = false; // set tag to show that box is closed again

thisElem = new objElem;
thisElem.name = tagcont[RowOut[row]][ColOut[col]];

// save procesdata
RecordEventData(thisElem, objEvent)

if (document.getElementById)  
	{
	// IE6/NS6>/Mozilla
	HandleTxt = document.getElementById(fieldname+"_txt");
	HandleBox = document.getElementById(fieldname+"_box");
	}
	else if (document.all)
	{
	//IE4/5
	HandleTxt=eval("document.all['"+fieldname+"_txt"+"']");
	HandleBox=eval("document.all['"+fieldname+"_box"+"']");
	}

//HandleTxt.style.visibility='hidden';HandleBox.style.visibility='visible';
}

function SwitchCont(fieldname, objEvent)
{
// special function for clicking tasks

if (!loaded) {return;} // do not open boxes when page is loading
if (!tmDirectStart & tmActive & tmCurTime==0) {startTmBar();}
if (tmTimeUp) {return;}

// check if a click on a link (A) occurred. this happens for example 
// when a mlweb A link gets a focus (due to clicking the box) and a subject presses enter
// this is to prevent enters from generating events when in click (rather than mouseover) mode

if (objEvent.srcElement)
	{
		if (objEvent.srcElement.nodeName=="A") {return}
	}
	else if (objEvent.target)
	{
		if (objEvent.target.nodeName=="A") {return}
	}
	
thisElem = new objElem
var row = abc_num(fieldname);
var col = parseInt(fieldname.substr(1));
// check if open cell should be recorded
if ((statecont[RowOut[row]][ColOut[col]]=="0") & !(recOpenCells)) {return;}

if (fieldname==prevfieldname)
	{
	// just close current box if box is same as previous	
	window.clearTimeout(delay);

	var row = abc_num(fieldname);
	var col = parseInt(fieldname.substr(1));
	thisElem.name = tagcont[RowOut[row]][ColOut[col]];

	// save procesdata
	RecordEventData(thisElem, objEvent)

	if (document.getElementById)  
		{
		// IE6/NS6>/Mozilla
		HandleTxt = document.getElementById(fieldname+"_txt");
		HandleBox = document.getElementById(fieldname+"_box");
		}
		else if (document.all)
		{	
		//IE4/5
		HandleTxt=eval("document.all['"+fieldname+"_txt"+"']");
		HandleBox=eval("document.all['"+fieldname+"_box"+"']");
		}

		HandleTxt.style.visibility='hidden';HandleBox.style.visibility='visible';
		prevfieldname="";
	}
	else
	{
	if ((prevfieldname!="")&(evtClose==1))
		{
		// first close prev box	if box is not same as previous
		window.clearTimeout(delay);
		var row = abc_num(prevfieldname);
		var col = parseInt(prevfieldname.substr(1));
		thisElem.name = tagcont[RowOut[row]][ColOut[col]];

		// save procesdata
		RecordEventData(thisElem, objEvent)

		if (document.getElementById)  
			{
			// IE6/NS6>/Mozilla
			HandleTxt = document.getElementById(prevfieldname+"_txt");
			HandleBox = document.getElementById(prevfieldname+"_box");
			}
			else if (document.all)
			{	
			//IE4/5
			HandleTxt=eval("document.all['"+prevfieldname+"_txt"+"']");
			HandleBox=eval("document.all['"+prevfieldname+"_box"+"']");
			}
	
			HandleTxt.style.visibility='hidden';HandleBox.style.visibility='visible';
		}
	
	if ((prevfieldname=="")|(evtClose==1))
	{
	// only if any box may be opened or there as no previous box open, show content
	var row = abc_num(fieldname);
	var col = parseInt(fieldname.substr(1));
	thisElem.name = tagcont[RowOut[row]][ColOut[col]];
	thisElem.value = txtcont[RowOut[row]][ColOut[col]];

	RecordEventData(thisElem, objEvent);

	if (document.getElementById)  
		{
		// IE6/NS6>/Mozilla
		HandleTxt = document.getElementById(fieldname+"_txt");
		HandleBox = document.getElementById(fieldname+"_box");
		}
		else if (document.all)
		{
		//IE4/5
		HandleTxt=eval("document.all['"+fieldname+"_txt"+"']");
		HandleBox=eval("document.all['"+fieldname+"_box"+"']");
		}
	
	// delay

	currCell = -1; 
	for (var i=0;i<Dlist.length;i++)
		{if (tagcont[RowOut[row]][ColOut[col]]==Dlist[i]) {currCell=i;break;}}
	
	if ((prevCell!=-1)&(currCell!=-1)) {dtime = DTimes[currCell][prevCell];} else {dtime=0}; 
	prevCell = currCell;
	
	//HandleTxt.style.visibility='visible';HandleBox.style.visibility='hidden';
	
	delay=window.setTimeout("HandleTxt.style.visibility='visible';HandleBox.style.visibility='hidden';",dtime)  //make image transparant
	prevfieldname=fieldname;
	}
	}
}


					
function recChoice(eventname ,name, value)
{
chkchoice = true;
timefunction(eventname, name, value);

if (document.forms[mlweb_fname].choice) {document.forms[mlweb_fname].choice.value = name;}

if (btnType=="button")
	{
	for (i=0;i<btnTxt.length;i++)
		{
		if (btnFlg==1) {btnNum = ColOut[i]} else {btnNum = RowOut[i]};
		HandleBut = eval("document.forms['"+mlweb_fname+"']."+btnTag[btnNum]);
		
		if (btnTag[btnNum]==name) {HandlePressed = HandleBut};
		
		if (btnState[btnNum]=="1") {HandleBut.className = 'btnStyle';}
		}
		
		HandlePressed.className='pressedStyle btnHov';
}
	
}
function loadMatrices()
{
// get settings data from script in body
txtcont = ExpMatrix(txt);
statecont = ExpMatrix(state);  
tagcont = ExpMatrix(tag);	
boxcont = ExpMatrix(box);
WidthCol = ExpRow(W_Col);
HeightRow = ExpRow(H_Row);
DTimes = ExpMatrix(delay);

CountCol = ExpRow(CBCol);
CountRow = ExpRow(CBRow);

btnTxt = ExpRow(btntxt);
btnTag = ExpRow(btntag);
btnState = ExpRow(btnstate);

// new in version 99.2: CB preset matrix
if (CBpreset) {
	CBorder = ExpMatrix(CBord);
}

ColOut = new Array();
for (var i=0; i<CountCol.length; i++)
{ColOut[i]=i;}

RowOut = new Array();
for (var i=0; i<CountRow.length; i++)
{RowOut[i]=i;}

Dlist = new Array();
for (j=0;j<RowOut.length;j++)
	{
	for (i=0;i<ColOut.length;i++)
		{
		if (statecont[j][i]=="1") {Dlist[Dlist.length]=tagcont[j][i];}
		}
	}
}

function reorder()
{
    loaded=true;
/*    
// if referer present (or other php/asp code) then get current hit number
if (document.cookie.indexOf("mlweb_subject=")!=-1)
		{
		subjstr=document.cookie;
		subject=subjstr.substr(subjstr.indexOf("mlweb_subject=")+14);
		}

if (document.cookie.indexOf("mlweb_condnum=")!=-1)
		{
		subjstr=document.cookie;
		subjnr=parseInt(subjstr.substr(subjstr.indexOf("mlweb_condnum=")+14));
		subjtype = "cookie";
		//alert(subjnr + " " + subjtype);
		}
		else 
		{
			if (typeof ref_cur_hit!="undefined")
				{subjnr = ref_cur_hit; subjtype = "header"}
				else
				{ 
				subjnr=-1; subjtype = "random";	
				}
		}

	// if subj nr turns out to be not a number, or randomizer is set to true then set it to randomize
	if (isNaN(subjnr)|randomOrder) {subjnr=-1; subjtype = "random";}

if (document.forms[mlweb_fname].condnum) {document.forms[mlweb_fname].condnum.value = subjnr;}
if (document.forms[mlweb_fname].expname) {document.forms[mlweb_fname].expname.value = expname;}
if (document.forms[mlweb_fname].nextURL) {document.forms[mlweb_fname].nextURL.value = nextURL;}
if (document.forms[mlweb_fname].subject) {document.forms[mlweb_fname].subject.value = subject;}
if (document.forms[mlweb_fname].to_email) {document.forms[mlweb_fname].to_email.value = to_email;}


if (CBpreset)
{
if (subjnr==-1) {subjnr=Math.floor(Math.random()*CBorder.length)}
// CB order is preset in a matrix 
curord = Math.floor(subjnr/masterCond) % CBorder.length;
cbcount=0;
for (var i=0; i<CountCol.length; i++)
	{
	ColOut[i]=parseInt(CBorder[curord][cbcount]);
	cbcount++;
	}


RowOut = new Array();

for (var i=0; i<CountRow.length; i++)	
	{
	RowOut[i]=parseInt(CBorder[curord][cbcount]);
	cbcount++
	}

}

else
{

// code if no prespecified CBorder

// retrieve position of counterbalance groups 

var cf=new Array()  // position of fixed cols
var c1=new Array()  // position of c1 cols

for (var i=0; i<CountCol.length; i++)
	{
	switch (CountCol[i])
		{ 
		case '0': cf[cf.length]=i;break;
		case '1': c1[c1.length]=i;break;
		}
	}

var rf=new Array()  // position of fixed rows
var r1=new Array()  // position of c1 rows

for (var i=0; i<CountRow.length; i++)
	{
	switch (CountRow[i])
		{ 
		case '0': rf[rf.length]=i;break;
		case '1': r1[r1.length]=i;break;
		}
	}

// subjDen is the denominator used to devide the subj number for each counterbalance step

subjDen = 1;   

if (subjtype!="random") {subjDen = Math.floor(subjDen * masterCond)};
// first determine column and row connects and switch on that

var numCond = (c1.length>0 ? fac(c1.length) : 1)*(r1.length>0 ? fac(r1.length) : 1);

if (subjnr==-1) {subjnr=Math.floor(Math.random()*numCond)}
//alert("total cond:" + numCond+"\nsubject: "+subjnr);

// counterbalance col groups		
if (c1.length>0) {c1_order=CountBal(subjnr/subjDen+1,c1.length); 
					subjDen = subjDen*fac(c1.length);} 

var c1count=0;
ColOut = new Array();

for (var i=0; i<CountCol.length; i++)
	{
	switch (CountCol[i])
		{ 
		case '0': ColOut[i]=i;break;
		case '1': ColOut[i]=c1[c1_order[c1count]];c1count++;break;
		}
	}

// counterbalance rows					
if (r1.length>0) {r1_order=CountBal(subjnr/subjDen+1,r1.length); subjDen = subjDen * fac(r1.length);} 

var r1count=0;
RowOut = new Array();

for (var i=0; i<CountRow.length; i++)
	{
	switch (CountRow[i])
		{ 
		case '0': RowOut[i]=i;break;
		case '1': RowOut[i]=r1[r1_order[r1count]];r1count++;break;
		}
	}
}

Dlist=new Array();

// reorder and resize table content
	for (j=0;j<RowOut.length;j++)
	{
	for (i=0;i<ColOut.length;i++)
		{
		var label = String.fromCharCode(j+97)+i.toString();
		if (statecont[j][i]=="1") {Dlist[Dlist.length]=tagcont[j][i];}

		if (document.getElementById)  
			{
			// IE6/NS6>/Mozilla
			HandleCont = document.getElementById(label+"_cont");
			HandleTxt = document.getElementById(label+"_txt");
			HandleBox = document.getElementById(label+"_box");
			HandleTD = document.getElementById(label+"_td");
			HandleTDbox = document.getElementById(label+"_tdbox");
			HandleImgBox = document.getElementById(label+"_img");
			HandleImg = eval("document.images."+label);
			pxstr="px";
			}
			else if (document.all)
			{
			//IE4/5
			HandleCont=eval("document.all['"+label+"_cont"+"']");
			HandleTxt=eval("document.all['"+label+"_txt"+"']");
			HandleTD=eval("document.all['"+label+"_td"+"']");
			HandleBox=eval("document.all['"+label+"_box"+"']");
			HandleTDbox=eval("document.all['"+label+"_tdbox"+"']");
			HandleImgbox=eval("document.all['"+label+"_img"+"']");
			HandleImg = eval("document.images."+label);
			pxstr="px";
			}
		
		// set txt 
		HandleTD.innerHTML =""; // empty for IE5 on mac bug
		// if txtcont is empty or only contains spaces then replace by nbsp to keep TD layout
		if (txtcont[RowOut[j]][ColOut[i]].replace(/[\x20]/gi, "")=="") {HandleTD.innerHTML = "&nbsp;"} 
			else {
					// if boxes are non-active and labels are fixed, header rows should also be fixed
					if (statecont[RowOut[j]][ColOut[i]]=="0")	
						{ 
						if (colFix) {tempcol = i} 
							else	{tempcol=ColOut[i]};

						if (rowFix) {temprow = j} 
							else	{temprow = RowOut[j]};

						HandleTD.innerHTML = txtcont[temprow][tempcol];
						}
				
					else {HandleTD.innerHTML = txtcont[RowOut[j]][ColOut[i]]};
			};

		HandleTDbox.innerHTML =""; // empty for IE5 on mac bug
		// if boxcont is empty or only contains spaces then replace by nbsp to keep TD layout
		if (boxcont[RowOut[j]][ColOut[i]].replace(/[\x20]/gi, "")=="") {HandleTDbox.innerHTML = "&nbsp;"} 
			else {
						if (colFix) {tempcol = i} 
							else	{tempcol=ColOut[i]};

						if (rowFix) {temprow = j} 
							else	{temprow = RowOut[j]};

						HandleTDbox.innerHTML = boxcont[temprow][tempcol];				
				};
		
		//set sizes
		HandleTD.width = parseInt(WidthCol[ColOut[i]])-5;
		HandleTD.height = parseInt(HeightRow[RowOut[j]])-5;
		HandleTDbox.width = parseInt(WidthCol[ColOut[i]])-5;
		HandleTDbox.height = parseInt(HeightRow[RowOut[j]])-5;
		if (statecont[RowOut[j]][ColOut[i]]=="1") {HandleTD.className = activeClass;} else {HandleTD.className = inactiveClass};
		HandleCont.style.width = parseInt(WidthCol[ColOut[i]])+pxstr;
		HandleCont.style.height = parseInt(HeightRow[RowOut[j]])+pxstr;
		HandleTxt.style.width = parseInt(WidthCol[ColOut[i]])+pxstr;
		HandleTxt.style.height = parseInt(HeightRow[RowOut[j]])+pxstr;
		HandleTxt.style.clip = "rect(0px "+ HandleTxt.style.width + " " + HandleTxt.style.height +" 0px)";
		HandleBox.style.width = parseInt(WidthCol[ColOut[i]])+pxstr;
		HandleBox.style.height = parseInt(HeightRow[RowOut[j]])+pxstr;
		HandleBox.style.clip = "rect(0px "+ HandleBox.style.width + " " + HandleBox.style.height +" 0px)";
		HandleImgBox.style.width = parseInt(WidthCol[ColOut[i]])+pxstr;
		HandleImgBox.style.height = parseInt(HeightRow[RowOut[j]])+pxstr;
		HandleImg.height = parseInt(HeightRow[RowOut[j]]);
		HandleImg.width	 = parseInt(WidthCol[ColOut[i]]);
		
		// open state=0 boxes using img names from imgcont matrix
		if (statecont[RowOut[j]][ColOut[i]] == '0')	{HandleBox.style.visibility = "hidden"; HandleTxt.style.visibility = "visible";} else {HandleBox.style.visibility = "visible"; HandleTxt.style.visibility = "hidden";}

		}
	}
	// if there are buttons then reorder the buttons according to the counterbalancing scheme
	if (btnFlg>0)
		{
		btn_inner = new Array()
		for (bc=0;bc<btnTxt.length;bc++)
			{				
			
			// swap names if not counterbalancing is turned off 
			if (btnFlg==1) 
				{
				//var btnNum = parseInt(ColOut[bc])
				realNum=parseInt(ColOut[bc]);
				if (colFix) {var txtNum = bc;  } 
					else   {var txtNum = realNum;} 
				}
				else {
				//var btnNum = parseInt(RowOut[bc])					
					realNum=parseInt(RowOut[bc]);
				if (rowFix) {var txtNum = bc;}
					else {var txtNum = realNum;}
					}
			if (btnState[realNum]=="1") 
						{
						if (btnType=="radio") 
							{var functionstr = "onMouseOver=\"timefunction('mouseover','"+btnTag[realNum]+"','"+btnTxt[realNum]+"')\" onClick=\"recChoice('onclick','"+btnTag[realNum]+"','"+btnTxt[txtNum]+"')\" onMouseOut=\"timefunction('mouseout','"+btnTag[realNum]+"','"+btnTxt[realNum]+"')\"";
							 	btn_inner[bc]="<INPUT type=\"radio\" name=\"mlchoice\" value=\""+btnTag[realNum]+"\" "+functionstr+">"+btnTxt[txtNum];}
							else
							{var functionstr = "onMouseOver=\"btnHover(this,'in');timefunction('mouseover','"+btnTag[realNum]+"','"+btnTxt[realNum]+"')\" onClick=\"recChoice('onclick','"+btnTag[realNum]+"','"+btnTxt[txtNum]+"')\" onMouseOut=\"btnHover(this,'out');timefunction('mouseout','"+btnTag[realNum]+"','"+btnTxt[realNum]+"')\"";
								btn_inner[bc]="<INPUT class=\"btnStyle\" type=\"button\" name=\"" + btnTag[realNum] + "\" value=\""+btnTxt[txtNum]+"\" "+functionstr+">";} 
						}
						else
						{btn_inner[bc]="&nbsp;";}
			
			}
			
			for (bc=0;bc<btnTxt.length;bc++)
				{	
					if (document.getElementById)  
						{
						// IE6/NS6>/Mozilla
						HandleTD = document.getElementById("btn_"+bc.toString());
						}
						else if (document.all)
							{
						//IE4/5
						HandleTD=eval("document.all['"+"btn_"+bc.toString()+"']");
						}
			if (bc==0) {defTDcolor = HandleTD.style.backgroundColor;}
			docstr=btn_inner[bc];
						
			HandleTD.innerHTML =""; // empty for IE5 on mac bug
			HandleTD.innerHTML = docstr;
				}
	}

// send col and row orders as events
timefunction("subject", subjtype,subjnr)
timefunction("order","col",ColOut.join("_"))
timefunction("order","row",RowOut.join("_"))
timefunction("events","open_close",evtOpen.toString()+"_"+evtClose.toString());
loaded = true; // set flag that page has been loaded;
if (tmActive) 	{
				initTmBar();
				if (tmDirectStart) {startTmBar();}
				}

return;
*/
}

function initTmBar() {
if (!tmActive) {return false;}

if (document.getElementById)  
	{
	// IE6/NS6>/Mozilla
	HandleTmCont = document.getElementById("tmCont");
	HandleTmBar = document.getElementById("tmBar");
	HandleTmTime = document.getElementById("tmTime");
	}
	else if (document.all)
	{
	//IE4/5
	HandleTmCont =eval("document.all['tmCont']");
	HandleTmBar=eval("document.all['tmBar']");
	HandleTmTime=eval("document.all['tmTime']");
	}
HandleTmCont.style.width=parseInt(tmWidthPx+4)+"px";
HandleTmTime.style.width=parseInt(tmWidthPx+4)+"px";

if (tmFill) {HandleTmBar.style.width="0px"; HandleTmTime.innerHTML="0 sec";} 
		else {HandleTmBar.style.width=parseInt(tmWidthPx)+"px"; HandleTmTime.innerHTML=parseInt(tmTotalSec)+" sec";}
if (tmShowTime) {HandleTmTime.style.visibility="visible";} 
		else {HandleTmTime.style.visibility="hidden";}
}

function startTmBar()
{
if (!tmActive) {return false;}
tmCurTime = 0;
tmInt = setInterval("refreshTmBar()", tmStepSec*1000);
}

function refreshTmBar()
{
if (!tmActive) {return false;}
tmCurTime= tmCurTime + tmStepSec*1000;
if (tmCurTime>tmTotalSec*1000) {clearInterval(tmInt); tmTimeUp=true; return;}
if (tmFill) {
			HandleTmBar.style.width=parseInt(Math.round(tmCurTime/(tmTotalSec*1000)*tmWidthPx))+"px";
			if (tmMinLabel=="false" | tmCurTime <60000) {HandleTmTime.innerHTML=parseInt(Math.round(tmCurTime/1000))+" "+tmSecLabel}
										else 
									{	var mnt = parseInt(Math.floor(tmCurTime/60000));
										var secs = parseInt((tmCurTime-mnt*60000)/1000);
										HandleTmTime.innerHTML=mnt+" "+tmMinLabel+" : "+secs+" "+tmSecLabel}	;
			}
			else {HandleTmBar.style.width=parseInt(tmWidthPx-Math.round(tmCurTime/(tmTotalSec*1000)*tmWidthPx))+"px";
			HandleTmTime.innerHTML=parseInt(tmTotalSec-Math.round(tmCurTime/1000))+" sec";
			if (tmMinLabel=="false" | tmCurTime<60000) {HandleTmTime.innerHTML=parseInt(tmTotalSec-Math.round(tmCurTime/1000))+" "+tmSecLabel}
										else 
									{	var timeleft = tmTotalSec*1000-tmCurTime;
										var mnt = parseInt(Math.floor(timeleft/60000));
										var secs = parseInt((timeleft-mnt*60000)/1000);
										HandleTmTime.innerHTML=mnt+" "+tmMinLabel+" : "+secs+" "+tmSecLabel}	;
			}
}
/////////////////////////
// end of mouselabcode //
/////////////////////////