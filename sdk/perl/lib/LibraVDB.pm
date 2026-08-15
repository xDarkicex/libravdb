package LibraVDB;
use strict;
use warnings;
use FFI::Platypus 2.00;
use Exporter 'import';

our @EXPORT_OK = qw(
    OpenDB
    CloseDB
    CreateCollection
    GetCollection
    InsertVector
    QueryVector
    ScanCollection
    Vacuum
    DropDatabase
    InsertBatch
    DeleteVector
    DeleteBatch
    DatabaseQuery
    DatabaseQueryWithParams
    FreeString
);

our %EXPORT_TAGS = ( all => \@EXPORT_OK );

# Initialize FFI
our $ffi = FFI::Platypus->new( api => 2 );

# Attempt to load the shared library
my $os = $^O;
my $lib_ext = $os eq 'darwin' ? 'dylib' : $os eq 'MSWin32' ? 'dll' : 'so';
my $lib_path = $ENV{LIBRAVDB_LIBRARY_PATH} || "../../cgo/libravdb.$lib_ext";

# Sometimes it's local in test mode
unless (-e $lib_path) {
    $lib_path = "./libravdb.$lib_ext";
}

$ffi->lib($lib_path);

# Attach core functions
$ffi->attach( 'OpenDB'           => ['string'] => 'int' );
$ffi->attach( 'CloseDB'          => ['int'] => 'void' );
$ffi->attach( 'CreateCollection' => ['int', 'string', 'int'] => 'int' );
$ffi->attach( 'GetCollection'    => ['int', 'string'] => 'int' );
$ffi->attach( 'InsertVector'     => ['int', 'string', 'float[]', 'int', 'string'] => 'string' );
$ffi->attach( 'QueryVector'      => ['int', 'float[]', 'int', 'int', 'string'] => 'string' );
$ffi->attach( 'ScanCollection'   => ['int', 'int', 'int'] => 'string' );
$ffi->attach( 'Vacuum'           => ['int'] => 'string' );
$ffi->attach( 'DropDatabase'     => ['int'] => 'string' );
$ffi->attach( 'InsertBatch'      => ['int', 'string[]', 'float[]', 'int', 'int', 'string[]'] => 'string' );
$ffi->attach( 'DeleteVector'     => ['int', 'string'] => 'string' );
$ffi->attach( 'DeleteBatch'      => ['int', 'string[]', 'int'] => 'string' );
$ffi->attach( 'DatabaseQuery'    => ['int', 'string'] => 'string' );
$ffi->attach( 'DatabaseQueryWithParams' => ['int', 'string', 'string'] => 'string' );
$ffi->attach( 'FreeString'       => ['opaque'] => 'void' );

1;
