%% https://tex.stackexchange.com/questions/638194/
\ExplSyntaxOn             % Switch on expl3 syntax
\sys_ensure_backend:      % load the backend code
\pdf_version_gset:n{1.7}  % Use provided expl3 function
\ExplSyntaxOff            % Switch off expl3 syntax

\LoadClass{scrartcl}
\RequirePackage[singlespacing=true]{scrlayer-scrpage}
\KOMAoptions{
	draft=off,
	fontsize=12pt,
	paper=letter,
	DIV=12,
	footnotes=multiple,
	numbers=withendperiod,
	headsepline=on,
	footsepline=off,
	abstract=on,
	headheight=36pt,
	footheight=36pt,
}
\RequirePackage[l2tabu, orthodox]{nag}

\RequirePackage{polyglossia}
\setdefaultlanguage[variant=american]{english}
% Appendix prefix for scrartcl
\newcommand*{\appendixmore}{%
	\renewcommand*{\sectionformat}{%
		\appendixname~\thesection\autodot\enskip%
	}%
	\renewcommand*{\sectionmarkformat}{%
		\appendixname~\thesection\autodot\enskip%
	}%
}

\RequirePackage[doublespacing]{setspace}
\RequirePackage[noorphans, listvskip, font=singlespacing]{quoting}
\RequirePackage{csquotes}
\AtBeginEnvironment{quoting}{\vspace{-1\baselineskip}}

\RequirePackage[en-US,calc]{datetime2}
\RequirePackage{fontawesome5}
\RequirePackage{etoolbox}
\RequirePackage{auxhook}
\RequirePackage{xurl}
\RequirePackage[table]{xcolor}
\RequirePackage{tikz}
\usetikzlibrary{backgrounds, calc, matrix, positioning}

\RequirePackage{titling}
\setlength{\droptitle}{-1.0in}

% Define CAP Economics official color
\definecolor{CAPEconPurple}{HTML}{200054}
% https://htmlcolorcodes.com/color-picker/
\definecolor{CAPEconPurpleTint4}{HTML}{9080aa}
\definecolor{CAPEconPurpleTint6}{HTML}{c7bfd4}
\definecolor{CAPEconPurpleTint7}{HTML}{e3dfea}

% Fonts and math
\RequirePackage{fontspec}
\setmonofont[Scale=MatchLowercase,]{Fira Mono}
\setsansfont{Fira Sans}
\setmainfont{STIX Two Text}

\RequirePackage[standard]{ntheorem}
\RequirePackage{mathtools}
\RequirePackage{unicode-math}
\setmathfont[math-style=ISO]{STIX Two Math}
% Separate \mathscr from \mathcal
\setmathfont{STIX Two Math}[range={scr,bfscr},StylisticSet=01]

\defaultfontfeatures[\rmfamily,\sffamily]{
	Ligatures={TeX, Common},
	Numbers={proportional, lining}
}
\RequirePackage[
	activate={true,nocompatibility},
	tracking=true,
]{microtype}

\RequirePackage{tabularray}
\UseTblrLibrary{booktabs}
%% Tables need sansserif font
%% https://tex.stackexchange.com/questions/286733/change-content-font-in-tabular-environments
\makeatletter
\appto\@floatboxreset{%
	\ifx\@captype\andy@table
		\sffamily\addfontfeatures{Numbers={Monospaced, Lining},}
	\fi
}
\def\andy@table{table}
\makeatother

%% Don't want full page figures to be vertically centered
%% https://tex.stackexchange.com/questions/28067/automatic-vertical-alignment-of-a-single-float-on-a-dedicated-page
\makeatletter
\setlength{\@fptop}{0pt}
\setlength{\@fpbot}{0pt plus 1fil}
\makeatother

%% Captions in Sans Serif
\addtokomafont{caption}{\sffamily}
\addtokomafont{caption}{\fontsize{10pt}{12pt}\selectfont}

%% Tweak footnotes
\setlength{\footnotesep}{1.5ex}
% \interfootnotelinepenalty=10000  % Force note text together
%% Read KOMA-Script documentation
\setfootnoterule[0.4pt]{\linewidth}
\deffootnote{2.5em}{2.5em}{%
	\makebox[2.5em][l]{\hspace{1ex}\thefootnotemark.}%
}

%% For highlighting text with a colored background, soul
\RequirePackage{soul}
\sethlcolor{yellow}

%% Properly typset intervals, and avoid messages about unmatched parentheses
\RequirePackage{interval}
\intervalconfig{
	soft open fences,
}

%%% custom commands, environments and symbols
\renewcommand{\ge}{\geqslant}\renewcommand{\geq}{\geqslant}
\renewcommand{\le}{\leqslant}\renewcommand{\leq}{\leqslant}
\DeclarePairedDelimiter\abs{\lvert}{\rvert}%
\DeclarePairedDelimiter\norm{\lVert}{\rVert}%

%% Define a paired-delimiter for defining sets
%% https://tex.stackexchange.com/questions/268766/curly-braces-in-math-mode
\DeclarePairedDelimiter\set\{\}
%% http://mirrors.ctan.org/macros/latex/contrib/mathtools/mathtools.pdf, 27--28
\DeclarePairedDelimiterX\SetSuchThat[2]{\lbrace}{\rbrace}{%
	#1 \, \delimsize\vert \, \mathopen{}#2
}

%% Sometimes, you just want to comment out several lies of text
%% https://tex.stackexchange.com/questions/44282/multiline-comment
\newcommand{\mycomment}[1]{}


%% Need a datetime package for various date related inserts
\RequirePackage[en-US,calc]{datetime2}

\RequirePackage{xurl}
\RequirePackage{hyperref}
\ProcessOptions\relax
\hypersetup{%
	unicode=true,
	bookmarksnumbered=false,
	hidelinks,
	pdfencoding=auto,
}
\RequirePackage[capitalize, noabbrev]{cleveref}

\AtBeginDocument{%
	\SetBlockEnvironment{quoting}
}

