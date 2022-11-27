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
    Dim x As Double
    Dim j As Long
    Dim i As Long
    Dim a As Long
    Dim xT As Long
    
    xT = 0
    a = 2
    x = 0
    i = 1
    j = 2
    
    While (j <= 226307)
    
        If (Sheet1.Cells(j, 8) = i) Then
            x = x + Sheet1.Cells(j, 3)
            xT = xT + 1
        Else
            x = x / xT
            xT = 0
            Sheet1.Cells(a, 10) = x
            Sheet1.Cells(a, 11) = i
            x = 0
            i = i + 1
            a = a + 1
        End If
        
        j = j + 1
    Wend
    
End Sub


