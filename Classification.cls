VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "Sheet1"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = True
Sub clc()
    Dim x As Long
    Dim j As Long
    Dim i As Long
    x = 1
    i = 2
    j = 3
    While (j <= 226307)
        If (Sheet1.Cells(i, 1) = Sheet1.Cells(j, 1)) Then
            Sheet1.Cells(i, 8) = x
            Sheet1.Cells(j, 8) = x
        Else
            x = x + 1
        End If
        i = i + 1
        j = j + 1
    Wend
End Sub
